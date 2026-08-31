import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rvc.nn import commons
from rvc.nn.discriminator import MultiPeriodDiscriminatorV2
from rvc.synthesizer import SynthesizerTrnMsNSFsid
from rvc.train.ckpt_utils import (
    checkpoints_dir,
    exported_epoch,
    export_model,
    latest_checkpoint_path,
    load_checkpoint,
    load_train_json,
    prune_keep_latest,
    save_checkpoint,
)
from rvc.runtime.paths import MODELS_DIR
from rvc.train.data_utils import BucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# 导出模型目录（统一来自 rvc.runtime.paths）；保留模块级名字，
# 云训练（autodl_train.py）靠运行时改 WEIGHTS_DIR 重定向到数据盘
WEIGHTS_DIR = MODELS_DIR


@dataclass
class TrainConfig:
    exp_dir: str
    sr: int = 48000
    epochs: int = 2000
    batch_size: int = 4
    save_every_epoch: int = 200
    learning_rate: float = 1e-4
    pretrain_g: str = ""
    pretrain_d: str = ""
    fp16_run: bool = True
    device: str = "cuda:0"
    log_interval: int = 20
    # checkpoint 含 optimizer 状态（G+D 约 0.7~0.8 GB/次），长期训练会撑爆磁盘；
    # 默认只留最新一组，够断点续训用。导出模型小得多，默认全部保留（0 = 不淘汰）。
    keep_ckpts: int = 1
    keep_models: int = 0


class Trainer:
    def __init__(self, train_config: TrainConfig, progress_callback=None, log_callback=None, loss_callback=None, batch_callback=None):
        self.cfg = train_config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.loss_callback = loss_callback
        self.batch_callback = batch_callback
        self.stop_requested = False
        self.json_config = load_train_json(self.cfg.sr)
        self.train_cfg = self.json_config["train"]
        self.data_cfg = self.json_config["data"]
        # use_spectral_norm 只属于判别器，不能混进生成器构造参数（基类虽有 **kwargs 吞掉，但保持干净）
        self.model_cfg = self.json_config["model"].copy()
        self.use_spectral_norm = bool(self.model_cfg.pop("use_spectral_norm", False))
        self.segment_size = self.train_cfg["segment_size"] // self.data_cfg["hop_length"]
        self.log_file = Path(self.cfg.exp_dir) / "train.log"

    def stop(self):
        self.stop_requested = True

    def cleanup(self):
        """释放 GPU 资源（用于训练停止后回收显存）"""
        for attr in ("synthesizer", "net_d", "optim_g", "optim_d", "scheduler_g", "scheduler_d"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def checkpoints_dir(self) -> Path:
        return checkpoints_dir(self.cfg.exp_dir)

    def log(self, message: str):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
        if self.log_callback:
            self.log_callback(message)

    def setup(self):
        torch.manual_seed(self.train_cfg.get("seed", 1234))
        random.seed(self.train_cfg.get("seed", 1234))
        np.random.seed(self.train_cfg.get("seed", 1234))
        spec_channels = self.data_cfg["filter_length"] // 2 + 1
        self.synthesizer = SynthesizerTrnMsNSFsid(
            spec_channels,
            self.segment_size,
            **self.model_cfg,
            is_half=self.cfg.fp16_run,
            sr=self.cfg.sr,
        ).to(self.cfg.device)
        self.net_d = MultiPeriodDiscriminatorV2(self.use_spectral_norm).to(self.cfg.device)
        self.optim_g = torch.optim.AdamW(self.synthesizer.parameters(), self.cfg.learning_rate, betas=self.train_cfg["betas"], eps=self.train_cfg["eps"])
        self.optim_d = torch.optim.AdamW(self.net_d.parameters(), self.cfg.learning_rate, betas=self.train_cfg["betas"], eps=self.train_cfg["eps"])
        self.start_epoch = 1

        # checkpoint 存在 <exp>/4_checkpoints/ 下（与 _save 一致）；
        # 根目录再试一次只为兼容早期布局，不改写入位置
        ckpt_dir = str(self.checkpoints_dir())
        latest_g = latest_checkpoint_path(ckpt_dir, "G") or latest_checkpoint_path(self.cfg.exp_dir, "G")
        latest_d = latest_checkpoint_path(ckpt_dir, "D") or latest_checkpoint_path(self.cfg.exp_dir, "D")
        if latest_g and latest_d:
            _, epoch_g = load_checkpoint(latest_g, self.synthesizer, self.optim_g)
            _, epoch_d = load_checkpoint(latest_d, self.net_d, self.optim_d)
            self.start_epoch = min(epoch_g, epoch_d) + 1
            self.log(f"恢复训练: epoch {self.start_epoch}（G={Path(latest_g).name} D={Path(latest_d).name}）")
        else:
            if self.cfg.pretrain_g:
                state = torch.load(self.cfg.pretrain_g, map_location="cpu", weights_only=False)
                self.synthesizer.load_state_dict(state.get("weight", state.get("model", state)), strict=False)
                self.log("加载预训练 G")
            if self.cfg.pretrain_d:
                state = torch.load(self.cfg.pretrain_d, map_location="cpu", weights_only=False)
                self.net_d.load_state_dict(state.get("weight", state.get("model", state)), strict=False)
                self.log("加载预训练 D")

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.train_cfg["lr_decay"], last_epoch=self.start_epoch - 2)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.train_cfg["lr_decay"], last_epoch=self.start_epoch - 2)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.fp16_run)

        filelist = str(Path(self.cfg.exp_dir) / "filelist.txt")
        dataset = TextAudioLoaderMultiNSFsid(filelist, self.data_cfg)
        sampler = BucketSampler(dataset, self.cfg.batch_size)
        self.loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, collate_fn=TextAudioCollateMultiNSFsid(), pin_memory=True)
        if len(self.loader) == 0:
            raise RuntimeError("训练样本不足，无法组成 batch")

    def train(self):
        self.setup()
        last_epoch = None
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            if self.stop_requested:
                break
            try:
                loss_g, loss_mel, loss_kl, loss_fm, loss_d = self._train_epoch(epoch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"显存不足（OOM）：请调小 Batch size（当前 {self.cfg.batch_size}）后重试。"
                    f"已清空缓存；若反复出现可同时减小 segment（配置于 assets/configs/ 的 json）"
                ) from None
            last_epoch = epoch
            self.scheduler_g.step()
            self.scheduler_d.step()

            # 每轮 loss 落盘（挂机复盘用，train.log 末尾追加）
            self.log(f"epoch {epoch:4d} | D {loss_d:.4f} | G {loss_g:.4f} | Mel {loss_mel:.4f} | KL {loss_kl:.4f} | FM {loss_fm:.4f}")

            if epoch % self.cfg.save_every_epoch == 0 or epoch == self.cfg.epochs or self.stop_requested:
                self._save(epoch)
            if self.progress_callback:
                self.progress_callback(epoch, self.cfg.epochs)
        if last_epoch is None:
            raise RuntimeError("训练在首个 epoch 前已停止")
        # 最终模型路径（已在 _save 中导出）
        return str(WEIGHTS_DIR / f"{Path(self.cfg.exp_dir).name}_e{last_epoch}.pth")

    def _train_epoch(self, epoch: int) -> tuple[float, float, float, float, float]:
        """训练一个 epoch，返回 (平均 G, Mel, KL, FM, D loss)"""
        self.synthesizer.train()
        self.net_d.train()
        total_batches = len(self.loader)
        g_sum = mel_sum = kl_sum = fm_sum = d_sum = 0.0
        loss_count = 0
        for batch_idx, batch in enumerate(self.loader, 1):
            if self.stop_requested:
                break
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = [x.to(self.cfg.device, non_blocking=True) for x in batch]
            wave = wave.unsqueeze(1)
            with torch.amp.autocast("cuda", enabled=self.cfg.fp16_run):
                y_hat, ids_slice, _, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.synthesizer(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(spec, self.data_cfg["filter_length"], self.data_cfg["n_mel_channels"], self.cfg.sr, self.data_cfg["mel_fmin"], self.data_cfg["mel_fmax"])
                y_mel = commons.slice_segments(mel, ids_slice, self.segment_size)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), self.data_cfg["filter_length"], self.data_cfg["n_mel_channels"], self.cfg.sr, self.data_cfg["hop_length"], self.data_cfg["win_length"], self.data_cfg["mel_fmin"], self.data_cfg["mel_fmax"])
                wave_slice = commons.slice_segments(wave, ids_slice * self.data_cfg["hop_length"], self.train_cfg["segment_size"])
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave_slice, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            self.optim_d.zero_grad(set_to_none=True)
            self.scaler.scale(loss_disc).backward()
            self.scaler.unscale_(self.optim_d)
            commons.clip_grad_value_(self.net_d.parameters(), None)  # 与上游一致：不裁剪（scaler 已处理 fp16）
            self.scaler.step(self.optim_d)

            with torch.amp.autocast("cuda", enabled=self.cfg.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave_slice, y_hat)
                loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * self.train_cfg["c_mel"]
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * self.train_cfg["c_kl"]
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            self.optim_g.zero_grad(set_to_none=True)
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(self.optim_g)
            commons.clip_grad_value_(self.synthesizer.parameters(), None)  # 与上游一致：不裁剪
            self.scaler.step(self.optim_g)
            self.scaler.update()

            if self.loss_callback:
                # 每 batch 都上报（GUI 侧自会节流刷新），不再受 log_interval 门槛限制——
                # 小数据集一个 epoch 可能不足 log_interval 个 batch，旧逻辑整个 epoch 都看不到 loss
                self.loss_callback({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss_d": float(loss_disc.detach().cpu()),
                    "loss_g": float(loss_gen_all.detach().cpu()),
                    "loss_mel": float(loss_mel.detach().cpu()),
                    "loss_kl": float(loss_kl.detach().cpu()),
                    "loss_fm": float(loss_fm.detach().cpu()),
                })

            if self.batch_callback:
                self.batch_callback(epoch, batch_idx, total_batches)

            g_sum += float(loss_gen_all.detach().cpu())
            mel_sum += float(loss_mel.detach().cpu())
            kl_sum += float(loss_kl.detach().cpu())
            fm_sum += float(loss_fm.detach().cpu())
            d_sum += float(loss_disc.detach().cpu())
            loss_count += 1

        n = max(loss_count, 1)
        return g_sum / n, mel_sum / n, kl_sum / n, fm_sum / n, d_sum / n

    def _save(self, epoch: int):
        """保存 checkpoint 并导出可用模型，随后按保留策略清理旧文件。"""
        ckpt_dir = self.checkpoints_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(self.synthesizer, self.optim_g, self.cfg.learning_rate, epoch, str(ckpt_dir / f"G_{epoch}.pth"))
        save_checkpoint(self.net_d, self.optim_d, self.cfg.learning_rate, epoch, str(ckpt_dir / f"D_{epoch}.pth"))
        self.log(f"保存 checkpoint: epoch {epoch}")

        # 只留最新 N 组（checkpoint 带 optimizer 状态，体积最大）
        keep_ckpts = max(self.cfg.keep_ckpts, 1)
        removed = []
        for prefix in ("G", "D"):
            removed += prune_keep_latest(ckpt_dir, f"{prefix}_*.pth", keep_ckpts)
        if removed:
            self.log(f"清理旧 checkpoint: {len(removed)} 个（每组保留最新 {keep_ckpts} 个）")

        # 同时导出可用模型到 assets/models/
        exp_name = Path(self.cfg.exp_dir).name
        output = WEIGHTS_DIR / f"{exp_name}_e{epoch}.pth"
        export_model(self.synthesizer.state_dict(), self.cfg.sr, self.json_config, epoch, str(output))
        self.log(f"导出模型: {output}")

        # 只清理本实验的 <exp>_e<N>.pth，不碰其他/合并出来的模型
        if self.cfg.keep_models > 0:
            gone = prune_keep_latest(WEIGHTS_DIR, f"{exp_name}_e*.pth", self.cfg.keep_models, epoch_of=exported_epoch)
            if gone:
                self.log(f"清理旧导出模型: {len(gone)} 个（保留最新 {self.cfg.keep_models} 个）")

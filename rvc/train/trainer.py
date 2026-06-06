import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rvc.nn import commons
from rvc.nn.discriminator import MultiPeriodDiscriminatorV2
from rvc.synthesizer import SynthesizerTrnMsNSFsid
from rvc.train.ckpt_utils import export_model, latest_checkpoint_path, load_checkpoint, load_train_json, save_checkpoint
from rvc.train.data_utils import BucketSampler, TextAudioCollateMultiNSFsid, TextAudioLoaderMultiNSFsid
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch


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


class Trainer:
    def __init__(self, train_config: TrainConfig, progress_callback=None, log_callback=None, loss_callback=None):
        self.cfg = train_config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.loss_callback = loss_callback
        self.stop_requested = False
        self.json_config = load_train_json(self.cfg.sr)
        self.train_cfg = self.json_config["train"]
        self.data_cfg = self.json_config["data"]
        self.model_cfg = self.json_config["model"].copy()
        self.segment_size = self.train_cfg["segment_size"] // self.data_cfg["hop_length"]

    def stop(self):
        self.stop_requested = True

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def setup(self):
        torch.manual_seed(self.train_cfg.get("seed", 1234))
        random.seed(self.train_cfg.get("seed", 1234))
        np.random.seed(self.train_cfg.get("seed", 1234))
        spec_channels = self.data_cfg["filter_length"] // 2 + 1
        self.net_g = SynthesizerTrnMsNSFsid(
            spec_channels,
            self.segment_size,
            **self.model_cfg,
            is_half=self.cfg.fp16_run,
            sr=self.cfg.sr,
        ).to(self.cfg.device)
        self.net_d = MultiPeriodDiscriminatorV2(self.model_cfg.get("use_spectral_norm", False)).to(self.cfg.device)
        self.optim_g = torch.optim.AdamW(self.net_g.parameters(), self.cfg.learning_rate, betas=self.train_cfg["betas"], eps=self.train_cfg["eps"])
        self.optim_d = torch.optim.AdamW(self.net_d.parameters(), self.cfg.learning_rate, betas=self.train_cfg["betas"], eps=self.train_cfg["eps"])
        self.start_epoch = 1

        latest_g = latest_checkpoint_path(self.cfg.exp_dir, "G")
        latest_d = latest_checkpoint_path(self.cfg.exp_dir, "D")
        if latest_g and latest_d:
            _, epoch_g = load_checkpoint(latest_g, self.net_g, self.optim_g)
            _, epoch_d = load_checkpoint(latest_d, self.net_d, self.optim_d)
            self.start_epoch = min(epoch_g, epoch_d) + 1
            self.log(f"恢复训练: epoch {self.start_epoch}")
        else:
            if self.cfg.pretrain_g:
                state = torch.load(self.cfg.pretrain_g, map_location="cpu", weights_only=False)
                self.net_g.load_state_dict(state.get("weight", state.get("model", state)), strict=False)
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
        self.loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=TextAudioCollateMultiNSFsid(), pin_memory=True)
        if len(self.loader) == 0:
            raise RuntimeError("训练样本不足，无法组成 batch")

    def train(self):
        self.setup()
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            if self.stop_requested:
                break
            self._train_epoch(epoch)
            self.scheduler_g.step()
            self.scheduler_d.step()
            if epoch % self.cfg.save_every_epoch == 0 or epoch == self.cfg.epochs or self.stop_requested:
                self._save(epoch)
            if self.progress_callback:
                self.progress_callback(epoch, self.cfg.epochs)
        output = Path("assets") / "weights" / f"{Path(self.cfg.exp_dir).name}.pth"
        export_model(self.net_g.state_dict(), self.cfg.sr, self.json_config, epoch, str(output))
        self.log(f"导出模型: {output}")
        return str(output)

    def _train_epoch(self, epoch: int):
        self.net_g.train()
        self.net_d.train()
        for batch_idx, batch in enumerate(self.loader, 1):
            if self.stop_requested:
                return
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = [x.to(self.cfg.device, non_blocking=True) for x in batch]
            wave = wave.unsqueeze(1)
            with torch.amp.autocast("cuda", enabled=self.cfg.fp16_run):
                y_hat, ids_slice, _, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                mel = spec_to_mel_torch(spec, self.data_cfg["filter_length"], self.data_cfg["n_mel_channels"], self.cfg.sr, self.data_cfg["mel_fmin"], self.data_cfg["mel_fmax"])
                y_mel = commons.slice_segments(mel, ids_slice, self.segment_size)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), self.data_cfg["filter_length"], self.data_cfg["n_mel_channels"], self.cfg.sr, self.data_cfg["hop_length"], self.data_cfg["win_length"], self.data_cfg["mel_fmin"], self.data_cfg["mel_fmax"])
                wave_slice = commons.slice_segments(wave, ids_slice * self.data_cfg["hop_length"], self.train_cfg["segment_size"])
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave_slice, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            self.optim_d.zero_grad(set_to_none=True)
            self.scaler.scale(loss_disc).backward()
            self.scaler.unscale_(self.optim_d)
            commons.clip_grad_value_(self.net_d.parameters(), None)
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
            commons.clip_grad_value_(self.net_g.parameters(), None)
            self.scaler.step(self.optim_g)
            self.scaler.update()

            if self.loss_callback and batch_idx % self.cfg.log_interval == 0:
                self.loss_callback({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss_d": float(loss_disc.detach().cpu()),
                    "loss_g": float(loss_gen_all.detach().cpu()),
                    "loss_mel": float(loss_mel.detach().cpu()),
                    "loss_kl": float(loss_kl.detach().cpu()),
                    "loss_fm": float(loss_fm.detach().cpu()),
                })

    def _save(self, epoch: int):
        save_checkpoint(self.net_g, self.optim_g, self.cfg.learning_rate, epoch, str(Path(self.cfg.exp_dir) / f"G_{epoch}.pth"))
        save_checkpoint(self.net_d, self.optim_d, self.cfg.learning_rate, epoch, str(Path(self.cfg.exp_dir) / f"D_{epoch}.pth"))
        self.log(f"保存 checkpoint: epoch {epoch}")

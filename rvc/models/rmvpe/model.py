"""RMVPE 模型 — F0 提取推理接口"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from rvc.models.rmvpe.transforms import MelSpectrogram
from rvc.models.rmvpe.blocks import E2E
from rvc.tools.cuda_graph import run_cuda_graph

logger = logging.getLogger(__name__)


class RMVPE:
    def __init__(self, model_path: str, is_half, device=None):
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)

        if str(self.device) == "cuda":
            self.device = torch.device("cuda:0")

        self.model = self._load_model(model_path, is_half)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        # cents 表与局部窗口偏移都常驻 GPU：解码必须在设备端完成，
        # 否则逐帧回 CPU 会在 sounddevice 回调里引入隐式同步（音频 glitch 源）。
        self.cents_mapping = torch.from_numpy(np.pad(cents_mapping, (4, 4))).float().to(self.device)
        self.local_offsets = torch.arange(9, device=self.device, dtype=torch.long)

    def _load_model(self, model_path: str, is_half: bool):
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()
        model = model.half() if is_half else model.float()
        return model.to(self.device)

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            mel = mel.half() if self.is_half else mel.float()
            hidden = run_cuda_graph(
                self.model, "rmvpe-network", lambda input_mel: self.model(input_mel), mel
            )
            return hidden[:, :n_frames, :]

    def decode(self, hidden, thred=0.03):
        """设备端解码 → f0 (T,) tensor。全程无 CPU 同步。"""
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * torch.pow(2.0, cents_pred / 1200.0)
        # cents==0 即被判为清音的帧（旧版靠 f0==10 反查置零）
        return torch.where(cents_pred > 0, f0, torch.zeros_like(f0))

    def infer_from_audio(self, audio, thred=0.03):
        """返回设备端 f0 tensor (T,) float32。

        调用方若需要 numpy（训练侧落盘）自行 .cpu().numpy()；推理侧全程留在 GPU。
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio_t = audio.float().to(self.device).unsqueeze(0)
        mel = run_cuda_graph(
            self.mel_extractor, "rmvpe-mel-extractor",
            lambda a: self.mel_extractor(a, center=True), audio_t
        )
        hidden = self.mel2hidden(mel).squeeze(0)
        # 解码是十几个小张量 kernel，launch 开销（0.3ms）远大于计算，
        # 单独包一层 CUDA Graph 后降到 0.06ms。thred 作为常量被捕获，故入 key。
        return run_cuda_graph(
            self.model, "rmvpe-decode-%s" % thred,
            lambda h: self.decode(h, thred=thred), hidden,
        )

    def to_local_average_cents(self, salience, thred=0.05):
        """局部加权平均解码（设备端，与旧 numpy 逐帧实现等价）。

        Args:
            salience: (T, 360) 网络输出（可 half/float）
            thred: 置信度阈值，峰值 ≤ 该值的帧判为清音

        Returns:
            cents: (T,) float32
        """
        if not torch.is_tensor(salience):
            salience = torch.from_numpy(salience)
        # 解码必须在 fp32 下做：cents 量级 2000~9000，half 的相对精度只有 ~1 音分
        salience = salience.to(self.device).float()
        n_frames = salience.shape[0]

        center = torch.argmax(salience, dim=1)                  # (T,)
        padded = F.pad(salience, (4, 4))                        # (T, 368)
        index = center.unsqueeze(1) + self.local_offsets        # (T, 9)
        local_salience = torch.gather(padded, 1, index)
        local_cents = torch.gather(
            self.cents_mapping.unsqueeze(0).expand(n_frames, -1), 1, index
        )
        divided = (local_salience * local_cents).sum(1) / local_salience.sum(1)
        maxx = salience.max(dim=1).values
        return torch.where(maxx > thred, divided, torch.zeros_like(divided))

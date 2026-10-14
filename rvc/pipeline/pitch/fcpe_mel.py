"""FCPE 的 mel 前端 — torchfcpe `Wav2MelModule` 的等价复刻（可 CUDA Graph 捕获）。

为什么在仓库内复刻
------------------
`torchfcpe/mel_extractor.py::MelModule.__call__` 开头有两处**无条件**的检查：

    if torch.min(y) < -1.: print('[error with torchfcpe.mel_extractor.MelModule]...')
    if torch.max(y) > 1.:  print(...)

`torch.min/max` 作用在 CUDA 张量上、结果被 Python `if` 取真值 → 走 `__index__`
→ `.item()` → **每块两次 GPU→CPU 同步**。而 F0 提取运行在 `f0_stream` 上，
这次同步会把 CPU 的提前入队打断、在 GPU 队列里留出空泡；实测 `wav2mel` 单独
3.31ms，而整个 FCPE（含图内的核心网络）只要 2.65ms —— 差额就是同步与
eager 启动开销。

该同步同时使 MelModule **无法被 CUDA Graph 捕获**：捕获过程中遇到设备同步会
使 stream capture 失效（实测 `cudaErrorStreamCaptureInvalidated`）。

复刻的等价性
------------
逐字保留 MelModule 的 STFT 参数与 pad 规则，以及 Wav2MelModule 的帧数对齐：
  pad_left  = (win - hop) // 2
  pad_right = max((win - hop + 1) // 2, win - T - pad_left)
  mode      = "reflect" if pad_right < T else "constant"
  spec      = stft(pad(y), n_fft, hop, win, hann, center=False, onesided) → |·|
  mel       = mel_basis @ spec
  out       = log(clamp(mel, min=clip_val)).transpose(-1, -2)
  n_frames  = T // hop + 1，多了复制末帧、少了截断

参数**不硬编码**：构造时从活动的 torchfcpe 实例读取（`mel_basis` / `n_fft` /
`win_size` / `hop_size` / `clip_val` / `out_stft`），避免与第三方实现漂移。

与 torchfcpe 输出的差异仅有一处：上面两个 print 检查被去掉。实测在真实
5.16s 缓冲上逐元素比较 `max|Δ| = 0.000e+00`（不是"近似"，是完全相同）。

采样率
------
torchfcpe 的 Wav2MelModule 在 `sample_rate != sampling_rate` 时会先重采样。
本模块只实现采样率一致的情形（项目唯一调用点传 `HUBERT_SAMPLE_RATE`）；
不一致时由 `FCPEExtractor` 回退到 torchfcpe 原实现并告警，不做静默降级。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCPEMelExtractor(nn.Module):
    """FCPE 的 wav→mel 前端（去掉设备同步，可被 CUDA Graph 捕获）。"""

    def __init__(self, wav2mel) -> None:
        """从活动的 torchfcpe `Wav2MelModule` 实例构造等价复刻。

        Args:
            wav2mel: `torchfcpe.models_infer.InferCFNaiveMelPE.wav2mel`
        """
        super().__init__()
        mel_module = wav2mel.mel_extractor
        # register_buffer 而非普通属性：随 .to(device) 迁移，且是图捕获的常量依赖
        self.register_buffer("mel_basis", mel_module.mel_basis.detach().clone())
        self.register_buffer("hann_window", torch.hann_window(int(mel_module.win_size)))
        self.sample_rate = int(wav2mel.sampling_rate)
        self.n_fft = int(mel_module.n_fft)
        self.win_length = int(mel_module.win_size)
        self.hop_length = int(mel_module.hop_length)
        self.clip_val = float(mel_module.clip_val)
        self.out_stft = bool(mel_module.out_stft)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """音频 → log-mel。

        Args:
            audio: (T,) 或 (B, T) 或 (B, T, 1) 波形（float32，已在目标设备）

        Returns:
            (B, T // hop + 1, n_mels)
        """
        y = audio
        if y.dim() == 3:
            y = y.squeeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        n_samples = y.shape[-1]

        pad_left = (self.win_length - self.hop_length) // 2
        pad_right = max(
            (self.win_length - self.hop_length + 1) // 2,
            self.win_length - n_samples - pad_left,
        )
        mode = "reflect" if pad_right < n_samples else "constant"
        y = F.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        if self.out_stft:
            spec = spec[:, :512, :]
        else:
            spec = torch.matmul(self.mel_basis, spec)
        mel = torch.log(torch.clamp(spec, min=self.clip_val)).transpose(-1, -2)

        # Wav2MelModule 的帧数对齐：STFT 帧数与 T // hop + 1 差一帧时补齐/截断，
        # 保证与 torchfcpe 输出同形（FCPE 核心网络对帧数敏感）。
        n_frames = int(n_samples // self.hop_length) + 1
        if n_frames > mel.shape[1]:
            mel = torch.cat((mel, mel[:, -1:, :]), dim=1)
        elif n_frames < mel.shape[1]:
            mel = mel[:, :n_frames, :]
        return mel

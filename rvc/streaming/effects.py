"""音频效果器 — RMS 混合 / SOLA / 麦克风气流声，统一 torch.Tensor（GPU）接口。

输出侧处理链：
  RMS 混合 → SOLA 拼接 → 麦克风气流声 → 硬件输出

所有效果器在 setup() 时初始化，热路径纯向量化（零 Python 循环）。
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from rvc.dsp.rms import apply_rms_mix
from rvc.dsp.sola import apply_sola

logger = logging.getLogger(__name__)

# 气流感噪声的固定种子：离线渲染逐样本可复现（实时链路对噪声序列无要求）
AIRFLOW_NOISE_SEED = 0x5EED_A17


class RmsMixEffect:
    """RMS 响度包络混合 — 让转换后音量参考原始音量。"""

    def __init__(self):
        self._hz_centis = 0

    def setup(self, sr: int):
        self._hz_centis = sr // 100

    def process(self, infer: torch.Tensor, ref: torch.Tensor, rms_mix: float) -> torch.Tensor:
        if rms_mix >= 1.0:
            return infer
        return apply_rms_mix(ref, infer, rms_mix, self._hz_centis)


class SolaEffect:
    """SOLA 时间对齐 — 消除块间不连续（相位对齐 + 交叉淡化）。"""

    def __init__(self):
        self.sola_buffer = None
        self._fade_in = None
        self._fade_out = None
        self._sola_norm_kernel = None
        self._block_samples = 0
        self._sola_buffer_samples = 0
        self._sola_search_samples = 0

    def setup(self, sr: int, block_samples: int, sola_buffer_samples: int,
              sola_search_samples: int, device: str):
        self._block_samples = block_samples
        self._sola_buffer_samples = sola_buffer_samples
        self._sola_search_samples = sola_search_samples
        self.sola_buffer = torch.zeros(self._sola_buffer_samples, device=device)
        ls = torch.linspace(0, 1, steps=self._sola_buffer_samples, device=device)
        self._fade_in = torch.sin(0.5 * np.pi * ls) ** 2
        self._fade_out = 1 - self._fade_in
        self._sola_norm_kernel = torch.ones(1, 1, self._sola_buffer_samples, device=device)

    def reset(self):
        if self.sola_buffer is not None:
            self.sola_buffer.zero_()

    def process(self, infer: torch.Tensor) -> torch.Tensor:
        return apply_sola(
            infer, self.sola_buffer, self._sola_norm_kernel,
            self._fade_in, self._fade_out,
            self._block_samples, self._sola_buffer_samples, self._sola_search_samples,
        )


class MicAirflowEffect:
    """麦克风气流声 — 自然化气流 + 人声融合。

    自然化（解决"太规律"）：
    - 辅音增强：高频能量比门控（p/b/t/s 时气流强，元音时弱），而非总能量
    - 非线性门控：pow(0.5)，中等音量也有明显气流，静音底噪不被压死
    - 呼吸 LFO：0.15Hz + 0.07Hz 双正弦，模拟呼吸周期的缓慢起伏
    - OU 随机游走：均值回复随机过程，模拟自然波动（0.5-2Hz 中速变化）
    - 随机突发：泊松过程平均每 6-7 秒一次意外喷麦（50-150ms，1.3-2x）

    人声融合（解决"太干净"）：
    - AM 幅度调制：气流超低频分量（15Hz 低通）对人声做 1.5% 深度调制，
      模拟气流冲击振膜导致人声输出被调制——融合感的核心
    - 近讲效应低频搁架：150Hz 以下提升，模拟指向性麦克风近距离物理特性
    - 气流控制软饱和：气流量控制软削波阈值（x/(1+|x|*drive)），
      气流大时人声微妙饱和（THD<1%），模拟振膜大位移非线性失真
    - 气流频谱跟随：高频嘶嘶声（2-4kHz 带通）量由人声高频能量比控制，
      辅音时气流也有高频成分，频谱特征与人声一致

    气流层：
    - 100Hz rumble（呼呼声主体，wind noise 能量集中在 <100Hz）
    - 75Hz resonance（vortex shedding 共振峰，WongWong 低沉共鸣）
    - 85Hz floor（静音稳态底噪）
    - 4kHz hiss（辅音高频嘶嘶声，喷麦感）

    所有低通均为因果卷积（只左侧pad），避免块末尾输出伸进右pad零区导致能量塌陷。
    块间上下文缓存保证块开头输出有完整历史，无V型凹陷。
    核大小随采样率动态缩放，时间常量/截止频率与sr无关。
    """

    # ── 增益（不依赖sr）──
    BREATH_GAIN = 5.4          # 呼呼声 rumble 增益（拉满时相对语音≈0dB）
    RESONANCE_GAIN = 0.5       # 共振层增益
    FLOOR_GAIN = 0.35          # 底噪增益（静音可闻，降低避免嘶滋）
    HISS_GAIN = 0.3            # 高频嘶嘶声增益（辅音时，受envelope_gate样本级门控）
    INTENSITY_SCALE = 0.35     # GUI 100% 对应实际 35% 效果

    # ── AM 幅度调制（人声融合核心）──
    AM_DEPTH = 0.04            # 4% 深度（气流包络样本级调制人声，气流大时调制更深）

    # ── 近讲效应（proximity effect）──
    PROXIMITY_GAIN = 0.4       # 低频提升量（约 +3dB @ 拉满）
    PROXIMITY_CUTOFF = 150     # Hz，低频搁架截止

    # ── 软饱和（气流控制）──
    SATURATION_BASE = 0.8      # 基础 drive（无时也有轻微饱和，模拟麦克风电路）
    SATURATION_AIR = 1.5       # 气流带来的额外 drive

    # ── 语音门控（单源定义）──
    GATE_MID = 0.08            # sigmoid 中点（RMS 包络量纲）
    GATE_STEEP = 30.0          # sigmoid 陡度
    GATE_EXP = 0.5             # 非线性门控指数（pow 0.5 = 平方根，中等音量更明显）
    GATE_EMA_FIRST = 0.08      # 门控 EMA 平滑权重
    GATE_EMA_DECAY = 0.92

    # ── 辅音增强 ──
    CONSONANT_BOOST = 1.8      # 辅音时气流增强倍数
    CONSONANT_HF_CUTOFF = 2000 # Hz，高频能量比计算的分界

    # ── 呼吸 LFO ──
    LFO1_FREQ = 0.15           # Hz，主呼吸周期
    LFO2_FREQ = 0.07           # Hz，次呼吸周期（非谐波叠加）
    LFO_AMP = 0.15             # LFO 幅度（±15%，减弱周期性让随机感更突出）

    # ── OU 随机游走（Ornstein-Uhlenbeck 过程）──
    OU_THETA = 0.02            # 均值回复速率（时间常数≈50s，比之前活跃4倍）
    OU_SIGMA = 0.08            # 噪声强度（翻倍，更大波动）
    OU_CLAMP = 0.4             # 钳位范围（±40%）

    # ── 随机突发（泊松过程）──
    BURST_PROB = 0.03          # 每块触发概率（@100ms块 = 平均每 3.3 秒一次，翻倍）
    BURST_DECAY = 0.05         # 突发衰减时间常数（秒）
    BURST_GAIN_MIN = 1.3       # 突发最小增益
    BURST_GAIN_MAX = 2.0       # 突发最大增益

    # @48k 参考核大小（setup中按sr比例缩放）
    _REF_SR = 48000
    _REF_SLOW_K = 2400         # 50ms box低通（包络/调制）
    _REF_HF_K = 480            # 10ms box低通（高频包络，快速响应辅音瞬态）
    _REF_FLOOR_K = 180         # 85Hz三角核
    _REF_BREATH_K = 150        # 100Hz三角核
    _REF_RESONANCE_K = 200     # 75Hz三角核
    _REF_HISS_K = 12           # 4kHz三角核（高频嘶嘶声低通）
    _REF_PROX_K = 51           # 150Hz box低通（近讲效应）
    _REF_GATE_LEN = 40         # 10ms EMA平滑

    @staticmethod
    def _triangular_kernel(k: int) -> torch.Tensor:
        """生成三角核（两次移动平均的卷积），频率响应sinc²，-26dB/octave。"""
        size = 2 * k - 1
        idx = torch.arange(size, dtype=torch.float32)
        w = torch.min(idx + 1, size - idx) / (k * k)
        return w.view(1, 1, size)

    def __init__(self):
        # 卷积核（CPU 模板 + GPU 缓存）
        self._slow_kernel = None
        self._combo_kernel = None
        self._gate_kernel = None
        self._prox_kernel = None
        self._slow_kernel_gpu = None
        self._combo_kernel_gpu = None
        self._gate_kernel_gpu = None
        self._prox_kernel_gpu = None
        # 上下文缓存长度
        self._slow_ctx = 0
        self._combo_ctx = 0
        self._gate_ctx = 0
        self._prox_ctx = 0
        # 预计算常量
        self._mod_norm = 1.0       # box低通后白噪声理论σ=sqrt(1/k)
        self._gate_offset = 0.0    # sigmoid(-2.4)≈0.0832
        self._lfo_step1 = 0.0      # LFO1 每样本相位增量
        self._lfo_step2 = 0.0      # LFO2 每样本相位增量
        self._am_alpha = 0.0       # AM 低通系数
        self._burst_decay_coef = 1.0  # 突发每样本衰减系数
        # 块间上下文缓存
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None
        self._prev_audio_prox = None
        # 慢变状态（每块更新）
        self._lfo_phase1 = 0.0
        self._lfo_phase2 = 0.0
        self._ou_state = 0.0
        self._burst_env = 0.0
        self._burst_gain = 0.0
        # 复用缓冲（热路径零分配）
        self._batch_in = None       # (2,1,ctx_n)：调制噪声/包络
        self._rng = None
        self._rng_device = None

    def setup(self, sr: int, device: str = None):
        """根据采样率动态计算所有核大小（时间常量/截止频率与sr无关）。"""
        scale = sr / self._REF_SR
        slow_k = max(1, int(self._REF_SLOW_K * scale))
        floor_k = max(1, int(self._REF_FLOOR_K * scale))
        breath_k = max(1, int(self._REF_BREATH_K * scale))
        resonance_k = max(1, int(self._REF_RESONANCE_K * scale))
        hiss_k = max(1, int(self._REF_HISS_K * scale))
        prox_k = max(1, int(self._REF_PROX_K * scale))
        hf_k = max(1, int(self._REF_HF_K * scale))
        gate_len = max(2, int(self._REF_GATE_LEN * scale))
        combo_size = 2 * max(floor_k, breath_k, resonance_k, hiss_k) - 1

        # slow：box 低通（包络 + 调制噪声）
        self._slow_kernel = torch.ones(1, 1, slow_k) / slow_k
        # hf：box 低通（高频包络，10ms快速响应）
        self._hf_kernel = torch.ones(1, 1, hf_k) / hf_k
        # prox：box 低通（近讲效应低频，短核）
        self._prox_kernel = torch.ones(1, 1, prox_k) / prox_k
        # combo：分组卷积组合核（4 通道：breath/resonance/floor/hiss）
        combo = torch.zeros(4, 1, combo_size)
        kb = self._triangular_kernel(breath_k)
        kr = self._triangular_kernel(resonance_k)
        kf = self._triangular_kernel(floor_k)
        kh = self._triangular_kernel(hiss_k)
        combo[0, 0, :kb.shape[2]] = kb[0, 0]
        combo[1, 0, :kr.shape[2]] = kr[0, 0]
        combo[2, 0, :kf.shape[2]] = kf[0, 0]
        combo[3, 0, :kh.shape[2]] = kh[0, 0]
        self._combo_kernel = combo
        # gate：指数 EMA（反转使最大权重对应最新样本）
        gate_kernel = self.GATE_EMA_FIRST * self.GATE_EMA_DECAY ** torch.arange(gate_len, dtype=torch.float32)
        self._gate_kernel = gate_kernel.flip(0).view(1, 1, -1)

        # 预计算常量
        self._slow_ctx = slow_k - 1
        self._combo_ctx = combo_size - 1
        self._gate_ctx = gate_len - 1
        self._prox_ctx = prox_k - 1
        self._mod_norm = (1.0 / slow_k) ** 0.5
        self._gate_offset = float(
            torch.sigmoid(torch.tensor(-self.GATE_MID * self.GATE_STEEP)).item()
        )
        # LFO 每样本相位增量
        self._lfo_step1 = 2.0 * np.pi * self.LFO1_FREQ / sr
        self._lfo_step2 = 2.0 * np.pi * self.LFO2_FREQ / sr
        # 突发衰减系数（每样本）
        self._burst_decay_coef = np.exp(-1.0 / (self.BURST_DECAY * sr))

        # 预迁移核到设备
        if device is not None:
            self._slow_kernel_gpu = self._slow_kernel.to(device)
            self._combo_kernel_gpu = self._combo_kernel.to(device)
            self._gate_kernel_gpu = self._gate_kernel.to(device)
            self._prox_kernel_gpu = self._prox_kernel.to(device)
            self._hf_kernel_gpu = self._hf_kernel.to(device)
        else:
            self._slow_kernel_gpu = None
            self._combo_kernel_gpu = None
            self._gate_kernel_gpu = None
            self._prox_kernel_gpu = None
            self._hf_kernel_gpu = None

    def reset(self):
        """重置所有有状态的缓存（warmup 后调用，确保离线从干净状态开始）。"""
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None
        self._prev_audio_prox = None
        self._lfo_phase1 = 0.0
        self._lfo_phase2 = 0.0
        self._ou_state = 0.0
        self._burst_env = 0.0
        self._burst_gain = 0.0
        self._rng = None
        self._rng_device = None

    def _noise(self, n: int, device: torch.device) -> torch.Tensor:
        """取 n 个标准正态噪声（私有 Generator，固定种子）。"""
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(AIRFLOW_NOISE_SEED)
            self._rng_device = device
        return torch.randn(n, device=device, generator=self._rng)

    @staticmethod
    def _causal_conv1d(x: torch.Tensor, kernel: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """因果1D卷积：只左侧pad(k-1)，输出长度=输入长度。"""
        k = kernel.shape[2]
        x_padded = F.pad(x, (k - 1, 0))
        return F.conv1d(x_padded, kernel, groups=groups)

    def process(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        # 旁路：强度 0 时不做任何处理，也不推进块间缓存。
        if intensity <= 0:
            return audio
        intensity = intensity * self.INTENSITY_SCALE

        n = audio.shape[0]
        device = audio.device

        # 确保核在当前 device 上
        if self._slow_kernel_gpu is None or self._slow_kernel_gpu.device != device:
            self._slow_kernel_gpu = self._slow_kernel.to(device)
            self._combo_kernel_gpu = self._combo_kernel.to(device)
            self._gate_kernel_gpu = self._gate_kernel.to(device)
            self._prox_kernel_gpu = self._prox_kernel.to(device)

        # ═══════════════════════════════════════════════════════════
        # 1. 包络 + 调制噪声（slow 卷积，2 通道）
        #    通道0=调制噪声, 通道1=|audio|(总包络)
        # ═══════════════════════════════════════════════════════════
        mod_raw = self._noise(n, device)
        if self._prev_audio is not None:
            audio_ctx = torch.cat([self._prev_audio, audio])
            mod_ctx = torch.cat([self._prev_mod, mod_raw])
        else:
            audio_ctx = audio
            mod_ctx = mod_raw
        self._prev_audio = audio_ctx[-self._slow_ctx:].clone()
        self._prev_mod = mod_ctx[-self._slow_ctx:].clone()

        ctx_n = mod_ctx.shape[0]
        batch_in = self._batch_in
        if batch_in is None or batch_in.shape[2] != ctx_n or batch_in.shape[0] != 2:
            batch_in = torch.empty(2, 1, ctx_n, device=device)
            self._batch_in = batch_in
        batch_in[0, 0].copy_(mod_ctx)
        torch.abs(audio_ctx, out=batch_in[1, 0])
        batch_out = self._causal_conv1d(batch_in, self._slow_kernel_gpu)
        mod_low = batch_out[0, 0, ctx_n - n:ctx_n]
        envelope = batch_out[1, 0, ctx_n - n:ctx_n]

        # 振幅调制归一化（理论σ，避免每块std估计跳变）
        mod_low = mod_low / self._mod_norm * 0.15 + 1.0  # MOD_DEPTH=0.15 用于 wind_mod
        mod_low = mod_low.clamp(0.7, 1.3)

        # ═══════════════════════════════════════════════════════════
        # 2. 样本级高频能量比（辅音检测）
        #    |diff| 经 10ms 快速 box 低通得到 hf_envelope（响应快，辅音→元音切换时10ms内降下来）
        #    hf_ratio = hf_envelope / envelope，样本级，辅音部分高、元音部分低
        #    彻底解决：①块级标量导致"块内有辅音则整块元音部分也出hiss"
        #             ②50ms慢低通导致"元音开头50ms仍有hiss残留"
        # ═══════════════════════════════════════════════════════════
        # 从拼接后的 audio_ctx 计算 |diff|（有完整跨块历史），前面补0对齐长度
        diff_ctx = torch.empty_like(audio_ctx)
        diff_ctx[0] = 0.0
        diff_ctx[1:] = torch.abs(audio_ctx[1:] - audio_ctx[:-1])
        hf_ctx_n = diff_ctx.shape[0]
        hf_envelope_full = self._causal_conv1d(
            diff_ctx.view(1, 1, -1), self._hf_kernel_gpu
        )[0, 0]
        hf_envelope = hf_envelope_full[hf_ctx_n - n:hf_ctx_n]
        hf_ratio = (hf_envelope / (envelope + 1e-8)).clamp(0.0, 1.0)

        # ═══════════════════════════════════════════════════════════
        # 3. 门控：sigmoid → 非线性 → 辅音增强 → EMA平滑
        # ═══════════════════════════════════════════════════════════
        gate_raw = torch.sigmoid((envelope - self.GATE_MID) * self.GATE_STEEP)
        gate_in = ((gate_raw - self._gate_offset) / (1.0 - self._gate_offset)).clamp(min=0.0)
        # 非线性门控：pow(0.5)，中等音量更明显，静音底噪不被压死
        gate_in = gate_in.pow(self.GATE_EXP)
        # 辅音增强：高频比高时气流增强（p/b/t/s 产生强气流）
        consonant_gain = 1.0 + (self.CONSONANT_BOOST - 1.0) * hf_ratio
        gate_in = gate_in * consonant_gain

        # 缓存平滑前的 gate（避免下一块二次滤波）
        if self._prev_gate is not None:
            gate_ctx = torch.cat([self._prev_gate, gate_in])
        else:
            gate_ctx = gate_in
        self._prev_gate = gate_ctx[-self._gate_ctx:].clone()
        gate_smooth = self._causal_conv1d(gate_ctx.view(1, 1, -1), self._gate_kernel_gpu)[0, 0]
        envelope_gate = gate_smooth[gate_ctx.shape[0] - n:]

        # ═══════════════════════════════════════════════════════════
        # 4. 呼吸 LFO（双正弦非谐波叠加，每块更新相位）
        # ═══════════════════════════════════════════════════════════
        t = torch.arange(n, device=device, dtype=torch.float32)
        lfo_val = 1.0 + self.LFO_AMP * 0.5 * (
            torch.sin(self._lfo_phase1 + t * self._lfo_step1) +
            0.6 * torch.sin(self._lfo_phase2 + t * self._lfo_step2 + 1.3)
        )
        self._lfo_phase1 = (self._lfo_phase1 + self._lfo_step1 * n) % (2.0 * np.pi)
        self._lfo_phase2 = (self._lfo_phase2 + self._lfo_step2 * n) % (2.0 * np.pi)

        # ═══════════════════════════════════════════════════════════
        # 5. OU 随机游走（均值回复，每块更新一次，线性插值）
        # ═══════════════════════════════════════════════════════════
        ou_rand = float(torch.randn(1, device=device, generator=self._rng).item())
        ou_new = self._ou_state + self.OU_THETA * (0.0 - self._ou_state) + self.OU_SIGMA * ou_rand
        ou_new = max(-self.OU_CLAMP, min(self.OU_CLAMP, ou_new))
        ou_val = self._ou_state + (ou_new - self._ou_state) * (t / n)
        self._ou_state = ou_new

        # ═══════════════════════════════════════════════════════════
        # 6. 随机突发（泊松过程 + 指数衰减包络）
        # ═══════════════════════════════════════════════════════════
        burst_rand = float(torch.rand(1, device=device, generator=self._rng).item())
        if burst_rand < self.BURST_PROB:
            self._burst_env = 1.0
            self._burst_gain = (self.BURST_GAIN_MIN +
                                 (self.BURST_GAIN_MAX - self.BURST_GAIN_MIN) *
                                 float(torch.rand(1, device=device, generator=self._rng).item()))
        burst_env = self._burst_env * (self._burst_decay_coef ** t)
        self._burst_env = float(burst_env[-1].item())
        burst_gain = 1.0 + (self._burst_gain - 1.0) * burst_env

        # 综合门控：envelope_gate × LFO × OU × 突发
        final_gate = envelope_gate * lfo_val * (1.0 + ou_val) * burst_gain

        # ═══════════════════════════════════════════════════════════
        # 7. 风噪声（combo 分组卷积，4 通道）
        # ═══════════════════════════════════════════════════════════
        noise = self._noise(n, device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise])
        else:
            noise_ctx = noise
        self._prev_noise = noise_ctx[-self._combo_ctx:].clone()

        noise_4ch = noise_ctx.view(1, 1, -1).expand(1, 4, -1)
        combo_out = self._causal_conv1d(noise_4ch, self._combo_kernel_gpu, groups=4)
        noise_n = noise_ctx.shape[0]
        breath_low = combo_out[0, 0, noise_n - n:noise_n]
        res_low = combo_out[0, 1, noise_n - n:noise_n]
        floor_low = combo_out[0, 2, noise_n - n:noise_n]
        hiss_low = combo_out[0, 3, noise_n - n:noise_n]

        # wind 缓慢起伏
        wind_mod = 1.0 + 0.4 * (mod_low - 1.0) / 0.15

        # 呼呼声 = rumble + resonance（综合门控）
        rumble = breath_low * final_gate * wind_mod * self.BREATH_GAIN
        resonance = res_low * final_gate * self.RESONANCE_GAIN
        breath = (rumble + resonance) * intensity

        # 底噪：稳态（不受门控影响）
        floor = floor_low * intensity * self.FLOOR_GAIN

        # 高频嘶嘶声：量由高频比控制（辅音时增强）+ envelope_gate 样本级门控（静音部分无嘶嘶）
        hiss = hiss_low * hf_ratio * envelope_gate * self.HISS_GAIN * intensity

        # ═══════════════════════════════════════════════════════════
        # 8. AM 幅度调制（气流包络样本级调制人声）—— 融合感核心
        #    直接用 envelope_gate（已10ms EMA平滑，截止约100Hz）作为样本级AM信号，
        #    气流大时调制更深，辅音起始瞬态也能触发调制
        # ═══════════════════════════════════════════════════════════
        am_factor = 1.0 + self.AM_DEPTH * envelope_gate * intensity
        audio_am = audio * am_factor

        # ═══════════════════════════════════════════════════════════
        # 9. 近讲效应低频搁架（150Hz 以下提升）
        # ═══════════════════════════════════════════════════════════
        if self._prev_audio_prox is not None:
            audio_prox_ctx = torch.cat([self._prev_audio_prox, audio_am])
        else:
            audio_prox_ctx = audio_am
        self._prev_audio_prox = audio_prox_ctx[-self._prox_ctx:].clone()
        audio_low = self._causal_conv1d(
            audio_prox_ctx.view(1, 1, -1), self._prox_kernel_gpu
        )[0, 0, audio_prox_ctx.shape[0] - n:]
        audio_prox = audio_am + audio_low * self.PROXIMITY_GAIN * intensity

        # ═══════════════════════════════════════════════════════════
        # 10. 气流控制软饱和（x / (1 + |x| * drive)）
        # ═══════════════════════════════════════════════════════════
        gate_mean = float(final_gate.mean().item())
        drive = self.SATURATION_BASE + gate_mean * self.SATURATION_AIR * intensity
        audio_sat = audio_prox / (1.0 + torch.abs(audio_prox) * drive)

        # ═══════════════════════════════════════════════════════════
        # 11. 混合输出：人声（AM+近讲+饱和）+ 呼呼声 + 底噪 + 嘶嘶声
        # ═══════════════════════════════════════════════════════════
        return audio_sat + breath + floor + hiss


class AudioProcessor:
    """音频处理编排器 — 持有 RMS / SOLA / 麦克风气流声。

    输出侧处理链：RMS 混合 → SOLA 拼接 → 麦克风气流声
    """

    def __init__(self):
        self.rms_mix = RmsMixEffect()
        self.sola = SolaEffect()
        self.airflow = MicAirflowEffect()

    def setup(self, sr: int, block_samples: int, sola_buffer_samples: int,
              sola_search_samples: int, device: str):
        self.rms_mix.setup(sr)
        self.sola.setup(sr, block_samples, sola_buffer_samples, sola_search_samples, device)
        self.airflow.setup(sr, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用）。"""
        self.sola.reset()
        self.airflow.reset()

    def process_output(self, infer, ref, rms_mix, airflow=0.0):
        """输出侧处理：RMS 混合 → SOLA → 麦克风气流声。"""
        infer = self.rms_mix.process(infer, ref, rms_mix)
        infer = self.sola.process(infer)
        infer = self.airflow.process(infer, airflow)
        return infer

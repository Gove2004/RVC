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
    """麦克风气流声 — v0 质感 + 停顿同步呼吸节奏（v6）。

    【裁决史】v0（2026-10-10，质感被认可：「还不如一开始的声音像」，
    但节奏不对）→ v1 事件驱动 / v2 低频触发 / v3 频响 FIR / v4 真实采样
    （「不要录入真实的音效，就用虚拟生成呼呼声。不要参考我的本地音频
    文件」）/ v5 湍流模型（「像呲呲声，不像呼呼声；又变成随机响了」）
    全部被否。

    【v6 = 恢复 v0 的全部质感层，只换节奏层】
    质感层（原样恢复，用户认可的「像」）：
      - 100Hz rumble（呼呼声主体）+ 75Hz 共振 + 85Hz 稳态底噪
      - 4kHz 辅音嘶声（样本级 hf_ratio 门控，辅音时才有）
      - AM 幅度调制（气流包络调制人声，融合感核心）
      - 近讲低频搁架 + 气流控制软饱和
    节奏层（唯一改动）：
      - 删 LFO 双正弦 / OU 随机游走 / 泊松突发——三者都与语音无关，
        「该响不响、不该响乱响」= 节奏不对的根源（v5 的「随机响」同理）
      - 改为**停顿同步的合成吸气 swell**：语音停顿（200ms 静音占比
        上升沿）触发一次纯合成隆起——指数 hold（τ=0.3s 衰减）经 150ms
        box 平滑成「快攻慢衰」的吸气包络，打在 rumble 层上。说话时
        沿用 v0 的语音门控（呼气耦合）。节奏跟语音走，零随机。

    块间一致性（硬规则，全部遵守）：
    - 缓存滤波**输入**而非输出（_prev_audio / _prev_mod / _prev_noise /
      _prev_gate / _prev_audio_prox / _prev_sil / _prev_swell）
    - 因果卷积只左侧 pad(k-1)；F.conv1d 是互相关，gate EMA 核已 flip
    - 噪声**每消费者独立流**（调制噪声 / 风噪各一个生成器 + 固定 16384
      补给池顺序消费）——单池多消费者的「值→消费者」映射随块长变化
      （v5 实测教训）；逐块 randn 值序也与调用长度相关
    - swell 链路全速率（不做抽取对齐）；hold 递推用 log 域 max-plus
      并行前缀扫描精确实现，跨块用 GPU 标量尾衔接
    - 零 .item() / Python 真值 / 0-dim 张量索引（GPU 标量只经张量算术）
    """

    # ── 增益（不依赖sr）──
    BREATH_GAIN = 5.4          # 呼呼声 rumble 增益（拉满时相对语音≈0dB）
    RESONANCE_GAIN = 0.5       # 共振层增益
    FLOOR_GAIN = 0.35          # 底噪增益（静音可闻，降低避免嘶滋）
    HISS_GAIN = 0.3            # 高频嘶嘶声增益（辅音时，受envelope_gate样本级门控）
    INTENSITY_SCALE = 0.35     # GUI 100% 对应实际 35% 效果

    # ── AM 幅度调制（人声融合核心）──
    AM_DEPTH = 0.04            # 4% 深度（气流包络样本级调制人声）

    # ── 近讲效应（proximity effect）──
    PROXIMITY_GAIN = 0.4       # 低频提升量（约 +3dB @ 拉满）

    # ── 软饱和（气流控制）──
    SATURATION_BASE = 0.8      # 基础 drive（无时也有轻微饱和，模拟麦克风电路）
    SATURATION_AIR = 1.5       # 气流带来的额外 drive

    # ── 语音门控（单源定义）──
    GATE_MID = 0.08            # sigmoid 中点（RMS 包络量纲）
    GATE_STEEP = 30.0          # sigmoid 陡度
    GATE_EXP = 0.5             # 非线性门控指数（中等音量更明显）
    GATE_EMA_FIRST = 0.08      # 门控 EMA 平滑权重
    GATE_EMA_DECAY = 0.92

    # ── 辅音增强 ──
    CONSONANT_BOOST = 1.8      # 辅音时气流增强倍数

    # ── 停顿同步呼吸 swell（节奏层，v6 新）──
    SILENCE_FLOOR = 0.015      # 包络低于此 = 静音
    SILENCE_SEC = 0.2          # 静音占比统计窗（秒）
    TRIG_LEVEL = 0.55          # 静音占比上升沿阈值（≈停顿 330ms 后触发）
    HOLD_TAU = 0.35            # 不应期 hold 时间常数（秒）
    HOLD_MIN = 0.15            # hold 低于此才允许新触发（防阈值附近抖动连发）
    SWELL_DECAY_TAU = 0.30     # swell 衰减时间常数（秒）
    SWELL_SMOOTH_SEC = 0.15    # swell 攻击平滑（box；快攻慢衰的「快攻」）
    SWELL_BOOST = 1.0          # swell 相对语音门控的增益

    _MAX_BLOCK_SEC = 1.0       # 内部子块上限（整段离线调用时切分；流式块远小于此）
    _NOISE_CHUNK = 16384       # 噪声池补给单位（CUDA randn 值序与调用块长相关）

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

    @staticmethod
    def _box(k: int) -> torch.Tensor:
        return torch.ones(1, 1, k) / k

    def __init__(self):
        # 卷积核（CPU 模板）+ GPU 缓存
        self._slow_kernel = None
        self._hf_kernel = None
        self._combo_kernel = None
        self._gate_kernel = None
        self._prox_kernel = None
        self._sil_kernel = None
        self._swell_kernel = None
        self._gpu = None
        # 预计算常量
        self._mod_norm = 1.0       # box低通后白噪声理论σ=sqrt(1/k)
        self._gate_offset = 0.0    # sigmoid(-2.4)≈0.0832
        self._sil_log_decay = 0.0  # 不应期 hold 每样本对数衰减系数
        self._swell_log_decay = 0.0  # swell hold 每样本对数衰减系数
        self._swell_norm = 1.0     # box 平滑对 hold 峰值的削低补偿
        # 上下文缓存长度
        self._slow_ctx = 0
        self._combo_ctx = 0
        self._gate_ctx = 0
        self._prox_ctx = 0
        self._hf_ctx = 0
        self._sil_ctx = 0
        self._swell_ctx = 0
        self._max_block = 0
        # 块间上下文缓存（存滤波输入而非输出）
        self._prev_audio = None       # 音频（slow 包络 2ch 批量卷积共用）
        self._prev_mod = None         # 调制噪声输入
        self._prev_noise = None       # 风噪输入（combo 4ch 共用）
        self._prev_gate = None        # 门控 EMA 的输入
        self._prev_audio_prox = None  # 近讲低通的输入
        self._prev_sil = None         # 静音指示流（全速率）
        self._prev_swell = None       # swell hold 的平滑输入
        # GPU 标量状态（跨块衔接；只经张量算术更新，绝不 .item()）
        self._ref_tail = None         # 不应期 hold 块尾
        self._swell_tail = None       # swell hold 块尾
        self._sil_tail = None         # 静音占比块尾（上升沿检测）
        self._samples_seen = 0        # 流起始守卫（纯 Python 计数）
        # 复用缓冲（热路径零分配）
        self._batch_in = None       # (2,1,ctx_n)：调制噪声/包络
        # 噪声流：每消费者独立（0=调制噪声, 1=风噪）
        self._rngs = [None, None]
        self._rng_device = None
        self._pools = [None, None]
        self._pool_pos = [0, 0]

    def setup(self, sr: int, device: str = None):
        """根据采样率动态计算所有核大小（时间常量/截止频率与sr无关）。"""
        scale = sr / self._REF_SR
        ref = self._REF_SR

        def k(ref_k: int) -> int:
            return max(1, int(ref_k * scale))

        slow_k, hf_k = k(self._REF_SLOW_K), k(self._REF_HF_K)
        floor_k, breath_k = k(self._REF_FLOOR_K), k(self._REF_BREATH_K)
        resonance_k, hiss_k = k(self._REF_RESONANCE_K), k(self._REF_HISS_K)
        prox_k, gate_len = k(self._REF_PROX_K), max(2, k(self._REF_GATE_LEN))
        sil_k = max(1, int(self.SILENCE_SEC * sr))
        smooth_k = max(1, int(self.SWELL_SMOOTH_SEC * sr))

        # slow：box 低通（包络 + 调制噪声）
        self._slow_kernel = self._box(slow_k)
        self._hf_kernel = self._box(hf_k)
        self._prox_kernel = self._box(prox_k)
        self._sil_kernel = self._box(sil_k)
        self._swell_kernel = self._box(smooth_k)
        # combo：分组卷积组合核（4 通道：breath/resonance/floor/hiss）
        combo_size = 2 * max(floor_k, breath_k, resonance_k, hiss_k) - 1
        combo = torch.zeros(4, 1, combo_size)
        for i, kern in enumerate((self._triangular_kernel(breath_k),
                                  self._triangular_kernel(resonance_k),
                                  self._triangular_kernel(floor_k),
                                  self._triangular_kernel(hiss_k))):
            combo[i, 0, :kern.shape[2]] = kern[0, 0]
        self._combo_kernel = combo
        # gate：指数 EMA（反转使最大权重对应最新样本）
        gate_kernel = self.GATE_EMA_FIRST * self.GATE_EMA_DECAY ** torch.arange(
            gate_len, dtype=torch.float32)
        self._gate_kernel = gate_kernel.flip(0).view(1, 1, -1)

        # 预计算常量
        self._slow_ctx = slow_k - 1
        self._combo_ctx = combo_size - 1
        self._gate_ctx = gate_len - 1
        self._prox_ctx = prox_k - 1
        self._hf_ctx = hf_k - 1
        self._sil_ctx = sil_k - 1
        self._swell_ctx = smooth_k - 1
        self._max_block = int(self._MAX_BLOCK_SEC * sr)
        self._mod_norm = (1.0 / slow_k) ** 0.5
        self._gate_offset = float(
            torch.sigmoid(torch.tensor(-self.GATE_MID * self.GATE_STEEP)).item()
        )
        # hold 每样本对数衰减（全速率，无抽取换算）
        self._sil_log_decay = -1.0 / (self.HOLD_TAU * sr)
        self._swell_log_decay = -1.0 / (self.SWELL_DECAY_TAU * sr)
        # box 平滑对脉冲 hold 峰值的削低：峰值 = τc/(L)·(1-exp(-L/τc))，
        # 补偿 = L / (τc·(1-exp(-L/τc)))（τc/L 为衰减长度量纲、样本）
        tc = self.SWELL_DECAY_TAU * sr
        self._swell_norm = smooth_k / (tc * (1.0 - float(np.exp(-smooth_k / tc))))
        self._gpu = None
        self.reset()

    def reset(self):
        """重置所有有状态的缓存（warmup 后调用，确保离线从干净状态开始）。"""
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None
        self._prev_audio_prox = None
        self._prev_sil = None
        self._prev_swell = None
        self._ref_tail = None
        self._swell_tail = None
        self._sil_tail = None
        self._samples_seen = 0
        self._rngs = [None, None]
        self._rng_device = None
        self._pools = [None, None]
        self._pool_pos = [0, 0]

    def _noise(self, stream: int, n: int, device: torch.device) -> torch.Tensor:
        """取第 stream 条噪声流的下 n 个标准正态值（独立生成器+固定补给池）。

        每条流独立种子、按固定 _NOISE_CHUNK 补给、顺序消费——单条流的值序
        与分块无关（CUDA randn 值序与单次调用长度相关，不能逐块 randn；
        多流共享单池则「值→消费者」映射与块长相关，也不能用）。
        """
        if self._rngs[stream] is None or self._rng_device != device:
            self._rngs[stream] = torch.Generator(device=device)
            self._rngs[stream].manual_seed(AIRFLOW_NOISE_SEED + stream)
            self._pools[stream] = torch.empty(0, device=device)
            self._pool_pos[stream] = 0
            self._rng_device = device
        pool, pos = self._pools[stream], self._pool_pos[stream]
        while pos + n > pool.numel():
            fresh = torch.randn(self._NOISE_CHUNK, device=device,
                                generator=self._rngs[stream])
            pool = torch.cat([pool[pos:], fresh])
            pos = 0
        out = pool[pos:pos + n]
        self._pools[stream] = pool
        self._pool_pos[stream] = pos + n
        return out

    @staticmethod
    def _causal_conv1d(x: torch.Tensor, kernel: torch.Tensor, groups: int = 1) -> torch.Tensor:
        """因果1D卷积：只左侧pad(k-1)，输出长度=输入长度。"""
        k = kernel.shape[2]
        x_padded = F.pad(x, (k - 1, 0))
        return F.conv1d(x_padded, kernel, groups=groups)

    def _scan_hold(self, x: torch.Tensor, tail, log_decay: float,
                   device, t: torch.Tensor) -> torch.Tensor:
        """精确递推 h[t] = max(x[t], h[t-1]·decay) 的 log 域 max-plus 并行前缀扫描。"""
        n = x.shape[0]
        ls = torch.log(x.clamp(min=1e-12))
        step = 1
        while step < n:
            add = step * log_decay
            ls = torch.cat([ls[:step], torch.maximum(ls[step:], ls[:-step] + add)])
            step <<= 1
        h = torch.exp(ls)
        if tail is not None and tail.device == device:
            h = torch.maximum(h, tail * torch.exp(t * log_decay))
        return h

    def _kernels_on(self, device: torch.device):
        """核的 GPU 缓存（首次/换设备时迁移）。"""
        if self._gpu is None or self._gpu[1] != device:
            self._gpu = (tuple(t.to(device) for t in (
                self._slow_kernel, self._hf_kernel, self._combo_kernel,
                self._gate_kernel, self._prox_kernel,
                self._sil_kernel, self._swell_kernel,
            )), device)
        return self._gpu[0]

    def process(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        # 旁路：强度 0 时不做任何处理，也不推进块间缓存。
        if intensity <= 0:
            return audio
        n_total = audio.shape[0]
        if n_total > self._max_block:
            # 整段离线调用：内部按 ≤1s 子块处理（上下文/状态跨子块衔接），
            # 保证与流式分块一致
            outs = [self._process_block(audio[i:i + self._max_block], intensity)
                    for i in range(0, n_total, self._max_block)]
            return torch.cat(outs)
        return self._process_block(audio, intensity)

    def _process_block(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        intensity = intensity * self.INTENSITY_SCALE
        n = audio.shape[0]
        device = audio.device
        k_slow, k_hf, k_combo, k_gate, k_prox, k_sil, k_swell = self._kernels_on(device)

        # ═══════════════════════════════════════════════════════════
        # 1. 包络 + 调制噪声（slow 卷积，2 样本批）
        #    batch 维 2：通道0=调制噪声, 通道1=|audio|(总包络)
        # ═══════════════════════════════════════════════════════════
        mod_raw = self._noise(0, n, device)
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
        batch_out = self._causal_conv1d(batch_in, k_slow)
        mod_low = batch_out[0, 0, ctx_n - n:ctx_n]
        envelope = batch_out[1, 0, ctx_n - n:ctx_n]

        # 振幅调制归一化（理论σ，避免每块std估计跳变）
        mod_low = mod_low / self._mod_norm * 0.15 + 1.0  # 0.15 为 wind_mod 的调制深度
        mod_low = mod_low.clamp(0.7, 1.3)

        # ═══════════════════════════════════════════════════════════
        # 2. 样本级高频能量比（辅音检测）
        #    hf_ratio = hf_envelope / envelope，样本级，辅音高、元音低
        # ═══════════════════════════════════════════════════════════
        diff_ctx = torch.empty_like(audio_ctx)
        diff_ctx[0] = 0.0
        diff_ctx[1:] = torch.abs(audio_ctx[1:] - audio_ctx[:-1])
        hf_ctx_n = diff_ctx.shape[0]
        hf_envelope_full = self._causal_conv1d(diff_ctx.view(1, 1, -1), k_hf)[0, 0]
        hf_envelope = hf_envelope_full[hf_ctx_n - n:hf_ctx_n]
        hf_ratio = (hf_envelope / (envelope + 1e-8)).clamp(0.0, 1.0)

        # ═══════════════════════════════════════════════════════════
        # 3. 停顿同步呼吸 swell（节奏层）：
        #    静音占比(200ms 窗)上升沿 → 不应期 hold（防抖）→ 指数 hold
        #    (τ=0.3s) → 150ms box 平滑 = 快攻慢衰的合成吸气包络。
        #    全速率链路（无抽取对齐问题）；跨块 GPU 标量尾衔接。
        # ═══════════════════════════════════════════════════════════
        sil_ind = (envelope < self.SILENCE_FLOOR).float()
        if self._prev_sil is not None:
            sil_ctx = torch.cat([self._prev_sil, sil_ind])
        else:
            sil_ctx = sil_ind
        self._prev_sil = sil_ctx[-self._sil_ctx:].clone()
        sil_v = self._causal_conv1d(sil_ctx.view(1, 1, -1), k_sil)[0, 0]
        sil_v = sil_v[sil_v.shape[0] - n:]

        if self._sil_tail is not None and self._sil_tail.device == device:
            head = self._sil_tail.view(1)
        else:
            head = torch.zeros(1, device=device)
        prev_v = torch.cat([head, sil_v[:-1]])
        self._sil_tail = sil_v[-1].detach()
        edge = (sil_v > self.TRIG_LEVEL).float() * (prev_v <= self.TRIG_LEVEL).float()

        t = torch.arange(n, device=device, dtype=torch.float32)
        # 不应期：hold(原始 edge) 衰减到 HOLD_MIN 以下才允许新触发
        h_ref = self._scan_hold(edge, self._ref_tail, self._sil_log_decay, device, t)
        if self._ref_tail is not None and self._ref_tail.device == device:
            h_prev = torch.cat([self._ref_tail.view(1), h_ref[:-1]])
        else:
            h_prev = torch.zeros(n, device=device)
        self._ref_tail = h_ref[-1].detach()
        edge_eff = edge * (h_prev < self.HOLD_MIN).float()

        # 流起始守卫：窗口未填满前不触发（冷启动静音不凭空吸气）
        if self._samples_seen < self._sil_ctx:
            n_invalid = min(n, self._sil_ctx - self._samples_seen)
            if n_invalid >= n:
                edge_eff = torch.zeros_like(edge_eff)
            else:
                edge_eff = edge_eff.clone()
                edge_eff[:n_invalid] = 0.0
        self._samples_seen += n

        # swell hold：τ=0.3s 指数衰减 → box 平滑（快攻慢衰）→ 峰值补偿
        swell_h = self._scan_hold(edge_eff, self._swell_tail,
                                  self._swell_log_decay, device, t)
        self._swell_tail = swell_h[-1].detach()
        if self._prev_swell is not None:
            swell_ctx = torch.cat([self._prev_swell, swell_h])
        else:
            swell_ctx = swell_h
        self._prev_swell = swell_ctx[-self._swell_ctx:].clone()
        swell = self._causal_conv1d(swell_ctx.view(1, 1, -1), k_swell)[0, 0]
        swell = swell[swell.shape[0] - n:] * self._swell_norm

        # ═══════════════════════════════════════════════════════════
        # 4. 门控：sigmoid → 非线性 → 辅音增强 → EMA平滑
        # ═══════════════════════════════════════════════════════════
        gate_raw = torch.sigmoid((envelope - self.GATE_MID) * self.GATE_STEEP)
        gate_in = ((gate_raw - self._gate_offset) / (1.0 - self._gate_offset)).clamp(min=0.0)
        gate_in = gate_in.pow(self.GATE_EXP)
        consonant_gain = 1.0 + (self.CONSONANT_BOOST - 1.0) * hf_ratio
        gate_in = gate_in * consonant_gain

        if self._prev_gate is not None:
            gate_ctx = torch.cat([self._prev_gate, gate_in])
        else:
            gate_ctx = gate_in
        self._prev_gate = gate_ctx[-self._gate_ctx:].clone()
        gate_smooth = self._causal_conv1d(gate_ctx.view(1, 1, -1), k_gate)[0, 0]
        envelope_gate = gate_smooth[gate_ctx.shape[0] - n:]

        # ═══════════════════════════════════════════════════════════
        # 5. 风噪声（combo 分组卷积，4 通道）
        # ═══════════════════════════════════════════════════════════
        noise = self._noise(1, n, device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise])
        else:
            noise_ctx = noise
        self._prev_noise = noise_ctx[-self._combo_ctx:].clone()

        noise_4ch = noise_ctx.view(1, 1, -1).expand(1, 4, -1)
        combo_out = self._causal_conv1d(noise_4ch, k_combo, groups=4)
        noise_n = noise_ctx.shape[0]
        breath_low = combo_out[0, 0, noise_n - n:noise_n]
        res_low = combo_out[0, 1, noise_n - n:noise_n]
        floor_low = combo_out[0, 2, noise_n - n:noise_n]
        hiss_low = combo_out[0, 3, noise_n - n:noise_n]

        # wind 缓慢起伏（调制噪声自调制）
        wind_mod = 1.0 + 0.4 * (mod_low - 1.0) / 0.15

        # 呼呼声 = rumble + resonance；门控 = 语音门控（呼气耦合）
        #          + 停顿吸气 swell（快攻慢衰，节奏跟语音走）
        gate_total = (envelope_gate + self.SWELL_BOOST * swell).clamp(max=1.8)
        rumble = breath_low * gate_total * wind_mod * self.BREATH_GAIN
        resonance = res_low * gate_total * self.RESONANCE_GAIN
        breath = (rumble + resonance) * intensity

        # 底噪：稳态（不受门控影响）
        floor = floor_low * intensity * self.FLOOR_GAIN

        # 高频嘶嘶声：量由高频比控制（辅音时增强）+ 样本级门控
        hiss = hiss_low * hf_ratio * envelope_gate * self.HISS_GAIN * intensity

        # ═══════════════════════════════════════════════════════════
        # 6. AM 幅度调制（气流包络样本级调制人声）—— 融合感核心
        # ═══════════════════════════════════════════════════════════
        am_factor = 1.0 + self.AM_DEPTH * envelope_gate * intensity
        audio_am = audio * am_factor

        # ═══════════════════════════════════════════════════════════
        # 7. 近讲效应低频搁架（150Hz 以下提升）
        # ═══════════════════════════════════════════════════════════
        if self._prev_audio_prox is not None:
            audio_prox_ctx = torch.cat([self._prev_audio_prox, audio_am])
        else:
            audio_prox_ctx = audio_am
        self._prev_audio_prox = audio_prox_ctx[-self._prox_ctx:].clone()
        audio_low = self._causal_conv1d(
            audio_prox_ctx.view(1, 1, -1), k_prox
        )[0, 0, audio_prox_ctx.shape[0] - n:]
        audio_prox = audio_am + audio_low * self.PROXIMITY_GAIN * intensity

        # ═══════════════════════════════════════════════════════════
        # 8. 气流控制软饱和（x / (1 + |x| · drive)，drive 逐样本跟随门控）
        #    ⚠️ 不能用块内均值（v0 旧法）：块长不同均值不同 → 整段与分块
        #    渲染不一致（实测 max|Δ| 2.7e-2）；样本级同时消除每块一次的
        #    drive 台阶
        # ═══════════════════════════════════════════════════════════
        drive = self.SATURATION_BASE + gate_total * (self.SATURATION_AIR * intensity)
        audio_sat = audio_prox / (1.0 + torch.abs(audio_prox) * drive)

        # ═══════════════════════════════════════════════════════════
        # 9. 混合输出：人声（AM+近讲+饱和）+ 呼呼声 + 底噪 + 嘶嘶声
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

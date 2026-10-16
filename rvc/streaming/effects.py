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
    """麦克风气流声 — v7 物理时序吸气 + 说话中呼吸退让（v10）。

    【裁决史】v0（2026-10-10，质感被认可：「还不如一开始的声音像」，
    但节奏不对）→ v1 事件驱动 / v2 低频触发 / v3 频响 FIR / v4 真实采样
    （「不要录入真实的音效，就用虚拟生成呼呼声。不要参考我的本地音频
    文件」）/ v5 湍流模型（「像呲呲声，不像呼呼声；又变成随机响了」）/
    v6 停顿触发 swell（「还是不太像，和人声的配合还是怪」）全部被否。
    v7（物理时序吸气）**被认可为新基线节点**；
    v8 吸气湍流带（250Hz–1kHz 带通噪声×吸气包络）被否——「说完话都有
    一个滋滋声，完全不可行」：带通白噪声无论怎么门控都是「滋」，不是
    气声（与 v5 呲呲声同病）；
    v9 尾呼气拖尾 + 门控保持 + 存气量深度被否——「基本上都是连续的呼呼
    隆隆声，偶尔说话有沙沙声」：①门控保持（τ=2.5s max-hold）让 rumble
    在停顿中不衰减、说话中又全速跟语音门控（含辅音 1.8× 突跳）→ 几乎
    整条时间线都在隆隆；②4kHz hiss 层（白噪声过宽带低通×样本级
    hf_ratio）本质就是宽带沙沙——转换语音自带辅音声，该层画蛇添足。
    **教训：呼吸声必须是离散事件，持续性增强方向（拖尾/保持）一律否决；
    说话中的合成嘶声/沙沙一律否决。**

    【v11 = v10 原样 + 存气量深度（v9 的深度项单独回融）】
    v10 被认可（「现在这个可以」）。真实转换统计：8/8 停顿有呼吸、
    吸气率 14.6 次/分（生理区间 14-20）——时序与密度已达标，短板是
    口气间深浅变化只由 wind_mod 随机起伏，没有因果。v11 只回融 v9
    三项中的存气量深度（持续项门控保持/尾呼气仍封死）：
      - air：说话中积累（τ=6s，语音门控驱动）、静音中缓慢消退（τ=12s）
      - depth = 0.35 + 0.65·clamp(air/0.35)——说得越久下一口气越深，
        短句后浅吸；一状态仿射（依赖语音门控而非吸气包络，无递归），
        前缀扫描精确、跨块标量尾衔接

    【v10 = v7 时序原样 + 两个结构性修正】
    质感层：
      - 100Hz rumble（呼呼声主体）+ 75Hz 共振 + 85Hz 稳态底噪
      - AM 幅度调制（气流包络调制人声，融合感核心）——不动
      - 近讲低频搁架 + 气流控制软饱和——不动
      - （v10 删）4kHz 辅音嘶声层：语音自带辅音，合成沙沙画蛇添足
    节奏层（v7 原样：物理时序、零事件触发、零随机）：
      - 静音持续 ≥0.12s 才开始吸气（词间闭止不误吸），τ=0.3s 升起，
        停顿越长吸得越足；静音 >1s 视为吸完，自然回落（τ=0.6s）；
        说话恢复 τ=0.05s 快速切断——「先吸气、后开口」的物理时序
      - 说话活跃度（τ=1s EMA）门控：长静音后不再呼吸，冷启动静音
        不凭空吸气
    呼吸退让（v10 新，VOICED_GATE_GAIN）：
      - 说话中 rumble 门控 ×0.4 退让（真实近讲：说话中的低频隆隆被
        语音掩蔽，呼吸声最突出的时刻是停顿吸气）——呼吸从「连续垫底」
        变「离散事件」；辅音 1.8× 突跳一并删除（起伏感来源）
      - 软饱和 drive 跟随语音门控原值（与 v7 说话中行为一致，人声
        质感不动）；AM 不受退让影响

    块间一致性（硬规则，全部遵守）：
    - 缓存滤波**输入**而非输出（_prev_audio / _prev_mod / _prev_noise /
      _prev_gate / _prev_audio_prox）
    - 因果卷积只左侧 pad(k-1)；F.conv1d 是互相关，gate EMA 核已 flip
    - 噪声**每消费者独立流**（调制噪声 / 风噪各一个生成器 + 固定 16384
      补给池顺序消费）——单池多消费者的「值→消费者」映射随块长变化
      （v5 实测教训）；逐块 randn 值序也与调用长度相关
    - dur / act / env / air 四条仿射递推（e'=a·e+b）用并行前缀扫描精确
      实现（组合律 (a_r,b_r)∘(a_l,b_l)=(a_l·a_r, a_r·b_l+b_r)），跨块用
      GPU 标量尾衔接——与分块无关
    - 软饱和 drive 逐样本跟随门控（v6 教训：块内均值随块长变）
    - 零 .item() / Python 真值 / 0-dim 张量索引（GPU 标量只经张量算术）
    """

    # ── 增益（不依赖sr）──
    BREATH_GAIN = 5.4          # 呼呼声 rumble 增益（拉满时相对语音≈0dB）
    RESONANCE_GAIN = 0.5       # 共振层增益
    FLOOR_GAIN = 0.35          # 底噪增益（静音可闻，降低避免嘶滋）
    INTENSITY_SCALE = 0.35     # GUI 100% 对应实际 35% 效果

    # ── 说话中呼吸退让（v10）：说话中 rumble 门控缩放——真实近讲里
    #    说话中的低频隆隆被语音掩蔽，呼吸声最突出的是停顿吸气。
    #    呼吸从「连续垫底」变「离散事件」；软饱和/AM 不受影响。──
    VOICED_GATE_GAIN = 0.4

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

    # ── 吸气包络（节奏层，v7）：物理时序，无事件触发 ──
    SILENCE_FLOOR = 0.015      # 包络低于此 = 静音
    INHALE_MIN_PAUSE = 0.12    # 静音不足此时长不吸气（词间闭止不误吸）
    INHALE_RISE_TAU = 0.30     # 吸气上升时间常数（停顿越长吸得越足）
    INHALE_RISE_END = 1.0      # 静音超过此时长 = 吸完了，转入回落
    INHALE_FALL_TAU = 0.60     # 长静音中吸气后的回落
    INHALE_RELEASE_TAU = 0.05  # 说话恢复时吸气快速切断（先吸气、后开口）
    INHALE_BOOST = 1.0         # 吸气相对语音门控的增益

    # ── 说话活跃度门控（v7）：长静音后不再呼吸、冷启动不凭空吸气 ──
    ACT_TAU = 1.0              # 活跃度 EMA 时间常数
    ACT_THRESHOLD = 0.05       # 活跃度归一阈值（≈刚可闻语音的包络）

    # ── 存气量深度（v11）：呼吸深浅由「最近说了多少话」决定——说得越
    #    久耗气越多，下一口气越深；短句后浅吸。生理因果的确定性变化
    #    （呼吸率 14.6 次/分已达标，缺的是口气间的深浅因果变化）──
    AIR_TAU_BUILD = 6.0        # 说话中存气量积累时间常数（消耗气体→需要吸气）
    AIR_TAU_DECAY = 12.0       # 静音中缓慢消退（呼吸本身会补一部分气）
    AIR_REF = 0.35             # air 归一参考（≈连续说话 3s 的积累量）
    INHALE_DEPTH_MIN = 0.35    # 存气量不足时的最浅吸气比例

    _MAX_BLOCK_SEC = 1.0       # 内部子块上限（整段离线调用时切分；流式块远小于此）
    _NOISE_CHUNK = 16384       # 噪声池补给单位（CUDA randn 值序与调用块长相关）

    # @48k 参考核大小（setup中按sr比例缩放）
    _REF_SR = 48000
    _REF_SLOW_K = 2400         # 50ms box低通（包络/调制）
    _REF_FLOOR_K = 180         # 85Hz三角核
    _REF_BREATH_K = 150        # 100Hz三角核
    _REF_RESONANCE_K = 200     # 75Hz三角核
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
        self._combo_kernel = None
        self._gate_kernel = None
        self._prox_kernel = None
        self._gpu = None
        # 预计算常量
        self._mod_norm = 1.0       # box低通后白噪声理论σ=sqrt(1/k)
        self._gate_offset = 0.0    # sigmoid(-2.4)≈0.0832
        # 吸气包络的仿射系数（setup 按 sr 计算）
        self._alpha_rise = 0.0     # 升起步进 α = 1-exp(-1/(τ·sr))
        self._rise_a = 1.0
        self._fall_a = 1.0
        self._rel_a = 1.0
        self._alpha_act = 0.0      # 活跃度 EMA 步进
        self._act_a = 1.0
        self._air_build_a = 1.0    # 存气量积累/消退系数
        self._air_decay_a = 1.0
        # 上下文缓存长度
        self._slow_ctx = 0
        self._combo_ctx = 0
        self._gate_ctx = 0
        self._prox_ctx = 0
        self._max_block = 0
        # 块间上下文缓存（存滤波输入而非输出）
        self._prev_audio = None       # 音频（slow 包络 2ch 批量卷积共用）
        self._prev_mod = None         # 调制噪声输入
        self._prev_noise = None       # 风噪输入（combo 3ch 共用）
        self._prev_gate = None        # 门控 EMA 的输入
        self._prev_audio_prox = None  # 近讲低通的输入
        # GPU 标量状态（跨块衔接；只经张量算术更新，绝不 .item()）
        self._dur_tail = None         # 静音持续时长块尾（样本）
        self._act_tail = None         # 说话活跃度块尾
        self._env_tail = None         # 吸气包络块尾
        self._air_tail = None         # 存气量块尾
        # 复用缓冲（热路径零分配）
        self._batch_in = None       # (2,1,ctx_n)：调制噪声/包络
        # 噪声流：每消费者独立（0=调制噪声, 1=风噪）
        self._rngs = [None, None]
        self._rng_device = None
        self._pools = [None, None]
        self._pool_pos = [0, 0]

    def setup(self, sr: int, device: str = None):
        """根据采样率动态计算所有核大小（时间常量/截止频率与sr无关）。"""
        self._sr = float(sr)
        scale = sr / self._REF_SR
        ref = self._REF_SR

        def k(ref_k: int) -> int:
            return max(1, int(ref_k * scale))

        slow_k = k(self._REF_SLOW_K)
        floor_k, breath_k = k(self._REF_FLOOR_K), k(self._REF_BREATH_K)
        resonance_k = k(self._REF_RESONANCE_K)
        prox_k, gate_len = k(self._REF_PROX_K), max(2, k(self._REF_GATE_LEN))

        # slow：box 低通（包络 + 调制噪声）
        self._slow_kernel = self._box(slow_k)
        self._prox_kernel = self._box(prox_k)
        # combo：分组卷积组合核（3 通道：breath/resonance/floor）
        combo_size = 2 * max(floor_k, breath_k, resonance_k) - 1
        combo = torch.zeros(3, 1, combo_size)
        for i, kern in enumerate((self._triangular_kernel(breath_k),
                                  self._triangular_kernel(resonance_k),
                                  self._triangular_kernel(floor_k))):
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
        self._max_block = int(self._MAX_BLOCK_SEC * sr)
        self._mod_norm = (1.0 / slow_k) ** 0.5
        self._gate_offset = float(
            torch.sigmoid(torch.tensor(-self.GATE_MID * self.GATE_STEEP)).item()
        )
        # 吸气包络仿射系数（e' = a·e + b 的逐步系数）
        self._alpha_rise = 1.0 - float(np.exp(-1.0 / (self.INHALE_RISE_TAU * sr)))
        self._rise_a = 1.0 - self._alpha_rise
        self._fall_a = float(np.exp(-1.0 / (self.INHALE_FALL_TAU * sr)))
        self._rel_a = float(np.exp(-1.0 / (self.INHALE_RELEASE_TAU * sr)))
        self._alpha_act = 1.0 - float(np.exp(-1.0 / (self.ACT_TAU * sr)))
        self._act_a = 1.0 - self._alpha_act
        self._air_build_a = float(np.exp(-1.0 / (self.AIR_TAU_BUILD * sr)))
        self._air_decay_a = float(np.exp(-1.0 / (self.AIR_TAU_DECAY * sr)))
        self._gpu = None
        self.reset()

    def reset(self):
        """重置所有有状态的缓存（warmup 后调用，确保离线从干净状态开始）。"""
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None
        self._prev_audio_prox = None
        self._dur_tail = None
        self._act_tail = None
        self._env_tail = None
        self._air_tail = None
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

    @staticmethod
    def _scan_affine(a: torch.Tensor, b: torch.Tensor, tail):
        """精确递推 e[i] = a[i]·e[i-1] + b[i] 的仿射并行前缀扫描。

        组合律：(a_l,b_l) 先、(a_r,b_r) 后 ⇒ (a_l·a_r, a_r·b_l + b_r)。
        返回完整序列 e[0..n-1]；tail 为上一块末状态 e[-1]（None=流起始）。
        与逐样本循环逐比特一致（浮点结合序相同），且与分块无关。
        """
        n = a.shape[0]
        A, B = a.clone(), b.clone()
        step = 1
        while step < n:
            a_r, b_r = A[step:], B[step:]
            a_l, b_l = A[:-step], B[:-step]
            # 先算 B（要用的旧 A），再更新 A
            B = torch.cat([B[:step], a_r * b_l + b_r])
            A = torch.cat([A[:step], a_l * a_r])
            step <<= 1
        if tail is not None and tail.device == a.device:
            return A * tail + B
        return B

    def _kernels_on(self, device: torch.device):
        """核的 GPU 缓存（首次/换设备时迁移）。"""
        if self._gpu is None or self._gpu[1] != device:
            self._gpu = (tuple(t.to(device) for t in (
                self._slow_kernel, self._combo_kernel,
                self._gate_kernel, self._prox_kernel,
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
        k_slow, k_combo, k_gate, k_prox = self._kernels_on(device)

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
        # 3. 吸气包络（节奏层，v7）：物理时序而非事件触发——
        #    静音中缓缓升起（≥0.12s 才起、τ=0.3s），峰值落在说话恢复的
        #    瞬间，语音一来快速切断（τ=0.05s）→「先吸气、后开口」；
        #    静音 >1s 视为吸完自然回落。活跃度门控在 4b（需语音包络）。
        #    dur / env / act 三条仿射递推并行前缀扫描精确实现，
        #    跨块 GPU 标量尾衔接——与分块无关。
        # ═══════════════════════════════════════════════════════════
        sil_ind = (envelope < self.SILENCE_FLOOR).float()

        # 3a. 静音持续时长（样本）：dur' = sil ? dur+1 : 0
        #     ⇒ 逐步系数 a=sil, b=sil（sil=0 时归零，sil=1 时线性累加）
        dur = self._scan_affine(sil_ind, sil_ind, self._dur_tail)
        self._dur_tail = dur[-1].detach()

        # 3b. 吸气包络：分段仿射系数（未满最短停顿持零 / 窗内升起 /
        #     超窗回落 / 有声快速切断）
        dur_s = dur * (1.0 / self._sr)
        early = (dur_s < self.INHALE_MIN_PAUSE).float()
        late = (dur_s > self.INHALE_RISE_END).float()
        in_win = (1.0 - early - late).clamp(min=0.0)
        a_env = (1.0 - sil_ind) * self._rel_a + sil_ind * (
            early + in_win * self._rise_a + late * self._fall_a)
        b_env = sil_ind * in_win * self._alpha_rise
        env_in = self._scan_affine(a_env, b_env, self._env_tail)
        self._env_tail = env_in[-1].detach()

        # ═══════════════════════════════════════════════════════════
        # 4. 门控：sigmoid → 非线性 → EMA平滑（无辅音增强——说话中
        #    rumble 随辅音 1.8× 突跳是「隆隆起伏感」来源，v10 删）
        # ═══════════════════════════════════════════════════════════
        gate_raw = torch.sigmoid((envelope - self.GATE_MID) * self.GATE_STEEP)
        gate_in = ((gate_raw - self._gate_offset) / (1.0 - self._gate_offset)).clamp(min=0.0)
        gate_in = gate_in.pow(self.GATE_EXP)

        if self._prev_gate is not None:
            gate_ctx = torch.cat([self._prev_gate, gate_in])
        else:
            gate_ctx = gate_in
        self._prev_gate = gate_ctx[-self._gate_ctx:].clone()
        gate_smooth = self._causal_conv1d(gate_ctx.view(1, 1, -1), k_gate)[0, 0]
        envelope_gate = gate_smooth[gate_ctx.shape[0] - n:]

        # ═══════════════════════════════════════════════════════════
        # 4b. 活跃度门控（v7）：τ=1s EMA of 包络——长静音后不再呼吸、
        #     冷启动不凭空吸气（act_n 平滑归零，无状态跳变）
        # ═══════════════════════════════════════════════════════════
        a_act = torch.full((n,), self._act_a, device=device)
        act = self._scan_affine(a_act, envelope * self._alpha_act, self._act_tail)
        self._act_tail = act[-1].detach()
        act_n = (act / self.ACT_THRESHOLD).clamp(0.0, 1.0)

        # 4c. 存气量深度（v11）：说话中积累（τ=6s，耗气→需要吸气）、
        #     静音中缓慢消退（τ=12s）。一状态仿射（依赖语音门控而非吸气
        #     包络，无递归），前缀扫描精确、跨块标量尾衔接。
        #     depth ∈ [0.35, 1.0]：说得越久下一口气越深，短句后浅吸——
        #     口气间深浅差异有因果，而非 wind_mod 的随机起伏。
        a_air = sil_ind * self._air_decay_a + (1.0 - sil_ind) * self._air_build_a
        b_air = (1.0 - sil_ind) * (1.0 - self._air_build_a) * envelope_gate
        air = self._scan_affine(a_air, b_air, self._air_tail)
        self._air_tail = air[-1].detach()
        depth = self.INHALE_DEPTH_MIN + (1.0 - self.INHALE_DEPTH_MIN) * \
            (air / self.AIR_REF).clamp(0.0, 1.0)
        inhale = env_in * act_n * depth

        # ═══════════════════════════════════════════════════════════
        # 5. 风噪声（combo 分组卷积，3 通道）
        # ═══════════════════════════════════════════════════════════
        noise = self._noise(1, n, device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise])
        else:
            noise_ctx = noise
        self._prev_noise = noise_ctx[-self._combo_ctx:].clone()

        noise_3ch = noise_ctx.view(1, 1, -1).expand(1, 3, -1)
        combo_out = self._causal_conv1d(noise_3ch, k_combo, groups=3)
        noise_n = noise_ctx.shape[0]
        breath_low = combo_out[0, 0, noise_n - n:noise_n]
        res_low = combo_out[0, 1, noise_n - n:noise_n]
        floor_low = combo_out[0, 2, noise_n - n:noise_n]

        # wind 缓慢起伏（调制噪声自调制）
        wind_mod = 1.0 + 0.4 * (mod_low - 1.0) / 0.15

        # 呼呼声 = rumble + resonance；门控 = 语音门控 × 退让（v10：
        #          说话中被语音掩蔽，呼吸退成离散事件）
        #          + 物理时序吸气包络（停顿中升起、开口瞬间切断）
        gate_total = (self.VOICED_GATE_GAIN * envelope_gate
                      + self.INHALE_BOOST * inhale).clamp(max=1.8)
        rumble = breath_low * gate_total * wind_mod * self.BREATH_GAIN
        resonance = res_low * gate_total * self.RESONANCE_GAIN
        breath = (rumble + resonance) * intensity

        # 底噪：稳态（不受门控影响）
        floor = floor_low * intensity * self.FLOOR_GAIN

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
        # 8. 气流控制软饱和（x / (1 + |x| · drive)，drive 逐样本跟随语音
        #    门控——与 v7 说话中行为一致（人声质感不动），不受退让影响。
        #    ⚠️ 不能用块内均值（v0 旧法）：块长不同均值不同 → 整段与分块
        #    渲染不一致（实测 max|Δ| 2.7e-2）
        # ═══════════════════════════════════════════════════════════
        drive = self.SATURATION_BASE + envelope_gate * (self.SATURATION_AIR * intensity)
        audio_sat = audio_prox / (1.0 + torch.abs(audio_prox) * drive)

        # ═══════════════════════════════════════════════════════════
        # 9. 混合输出：人声（AM+近讲+饱和）+ 呼呼声 + 底噪
        # ═══════════════════════════════════════════════════════════
        return audio_sat + breath + floor


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

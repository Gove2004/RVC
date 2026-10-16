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
    """麦克风气流声 — 纯合成湍流「呼呼」（v5）。

    【裁决史】v1 事件驱动合成（「和人声独立，假」）→ v2 低频带触发
    （「莫名其妙响起来 + 声音不像」）→ v3 实测频响 FIR（「跟频率没关系，
    完全不像」）→ v4 真实呼吸采样回放（「不要录入真实的音效，就用虚拟
    生成呼呼声。不要参考我的本地音频文件」）→ v5 纯合成。

    【物理机制】对麦说话的「呼呼」= 口鼻气流在振膜附近的湍流噪声。
    三要素全在时间域（不做频响雕塑——v3 教训）：

      1. 载波 carrier：宽带噪声经声学路径级联低通（暗色 rumble），RMS 归一
      2. 阵性起伏 gust：湍流强度本身在 ~20Hz 与 ~2Hz 两个尺度上混沌波动。
         用「慢速低通噪声的 |·|」（均值归一）当起伏函数——与载波同源、
         自相似的混沌信号。v1-v3 的人工包络 / box(|white|) 数学缺陷：
         大窗平均方差 ∝ 1/k，几乎恒定 → 恒定嘶声，这就是「假」的根源
      3. 气流驱动 drive：说话越响气流越大——**连续耦合**语音慢包络
         （200ms box 幂压缩）+ 恒定基流。无离散触发、无事件检测，
         电平永远连续（v1「独立」、v2「莫名响起」的解）
      4. 气流越大频谱越宽（湍流噪声 U^6 律）：带通层（box − 载波）
         随 drive² 增亮

    结构（v4 的采样库/停顿检测/状态机全部删除）：
      breath = gust · drive · G · (carrier + band · drive · BAND_GAIN)
      voice  = audio · (1 + 近讲搁架)
      output = voice + breath + 稳态底噪

    块间一致性（硬规则，全部遵守）：
    - 全链路无反馈状态机，只有「上下文缓存 + 顺序噪声池」，任意分块 ==
      整段（FP 卷积累加序差异在 1.5e-3 内）
    - 缓存滤波**输入**而非输出（_prev_audio / _prev_car / _prev_flick /
      _prev_noise 存各自滤波的原始输入）
    - 因果卷积只左侧 pad(k-1)
    - 噪声按**每消费者独立流**供给（载波/gust/底噪各一个生成器+池）。
      ⚠️ 不能多消费者共享一条顺序流：单池顺序消费下「哪段值分给哪条
      消费者」取决于块长（整段 1s 切 vs 随机分块映射不同），逐块 randn
      则值序本身与调用长度相关——两种都会破坏块间一致性。独立流各自
      按固定块长补给，消费只做顺序切片，与块长完全无关
    - 零 .item() / Python 真值 / 0-dim 张量索引
    """

    # ── 增益（不依赖 sr）──
    INTENSITY_SCALE = 0.35     # GUI 100% 对应实际 35% 效果
    BREATH_GAIN = 0.10         # 呼吸层总增益（载波/gust/带通均已 RMS 归一）
    BAND_GAIN = 0.30           # 带通层亮度（×drive²，气流大→频谱宽）
    FLOOR_GAIN = 0.35          # 稳态底噪增益
    PROXIMITY_GAIN = 0.4       # 近讲低频提升量

    # ── 气流驱动 drive（连续耦合语音包络）──
    DRIVE_BASE = 0.35          # 静默基流（房间空气连续感，不归零）
    DRIVE_SPEECH = 0.5         # 语音耦合分量
    DRIVE_REF = 0.08           # 语音包络参考（|x| 均值≈0.8·RMS，典型语音 RMS~0.1）
    DRIVE_MAX = 1.4            # 压缩上限
    DRIVE_EXP = 0.7            # 幂压缩（响度感知近似）

    # ── 阵性起伏 gust ──
    GUST_CEIL = 4.0            # 软上限：gust = ceil·tanh(f1·f2/ceil)

    # @48k 参考核大小（setup 中按 sr 比例缩放）
    _REF_SR = 48000
    _REF_DRIVE_K = 9600        # 200ms 语音慢包络 box
    _REF_PROX_K = 51           # 近讲低频 box
    _REF_FLOOR_K = 180         # 85Hz 三角核（稳态底噪）
    _REF_CAR_K1 = 64           # 载波级联 box 内层（null 750Hz）
    _REF_CAR_K2 = 160          # 载波级联 box 外层（null 300Hz）
    _REF_BAND_K = 48           # 带通 = box48 − 载波（300~750Hz 气息亮层）
    _REF_LP1_K = 800           # gust 快尺度级联 box（~20Hz 带宽）
    _REF_LP2_K = 7200          # gust 慢尺度级联 box（~2Hz 带宽，呼吸起伏）
    _INV_SQRT2PI = 0.7978845608  # E|N(0,1)| = sqrt(2/π)，gust 均值归一用

    _MAX_BLOCK_SEC = 1.0       # 内部子块上限（整段离线调用时切分；流式块远小于此）
    _NOISE_CHUNK = 16384       # 噪声池补给单位（CUDA randn 值序与调用块长相关）

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
        # FIR 核（CPU 模板）+ 归一常数（||h||₂，setup 时一次算好）
        self._k_drive = self._k_prox = self._k_floor = None
        self._k_car = self._k_band = self._k_lp1 = self._k_lp2 = None
        self._norm_car = self._norm_band = self._norm_lp1 = self._norm_lp2 = 1.0
        self._gpu = None  # (核元组, device) 缓存
        # 各滤波的输入尾缓存（存输入而非输出，防下一块二次滤波）
        self._prev_audio = None    # 原始音频（drive 包络 + 近讲共用）
        self._prev_car = None      # 载波/带通噪声输入
        self._prev_flick = None    # gust 噪声输入
        self._prev_noise = None    # 底噪输入
        # 上下文缓存长度
        self._a_ctx = 0
        self._car_ctx = 0
        self._flick_ctx = 0
        self._noise_ctx = 0
        self._max_block = 0
        # 噪声流：每消费者独立（0=载波/带通, 1=gust, 2=底噪）
        self._rngs = [None, None, None]
        self._rng_device = None
        self._pools = [None, None, None]
        self._pool_pos = [0, 0, 0]

    def setup(self, sr: int, device: str = None):
        """根据采样率动态计算所有 FIR（时间常量/截止频率与 sr 无关）。"""
        scale = sr / self._REF_SR
        ref = self._REF_SR

        def k(ref_k: int) -> int:
            return max(1, int(ref_k * scale))

        drive_k, prox_k, floor_k = k(self._REF_DRIVE_K), k(self._REF_PROX_K), k(self._REF_FLOOR_K)
        car1, car2, band_k = k(self._REF_CAR_K1), k(self._REF_CAR_K2), k(self._REF_BAND_K)
        lp1, lp2 = k(self._REF_LP1_K), k(self._REF_LP2_K)

        # 载波：box(car1) → box(car2) 级联（合并成单 FIR）。
        # ⚠️ F.conv1d 默认 valid 模式输出 k2−k1+1，级联 FIR 必须全 padding：
        # 输出长度 = k1+k2−1
        h_car = F.conv1d(self._box(car2), self._box(car1), padding=car1 - 1)
        # 带通：box(band_k) − 载波。合并成单 FIR 需右对齐（因果卷积核尾对齐当前样本）
        L = h_car.shape[2]
        h_band = F.pad(self._box(band_k), (L - band_k, 0)) - h_car
        # gust 两个尺度的低通（级联合并，全 padding 同上）
        h_lp1 = F.conv1d(self._box(lp1), self._box(lp1), padding=lp1 - 1)
        h_lp2 = F.conv1d(self._box(lp2), self._box(lp2), padding=lp2 - 1)

        self._k_drive = self._box(drive_k)
        self._k_prox = self._box(prox_k)
        self._k_floor = self._triangular_kernel(floor_k)
        self._k_car, self._k_band = h_car, h_band
        self._k_lp1, self._k_lp2 = h_lp1, h_lp2

        # RMS 归一常数：白噪声经 FIR h 后标准差 = ||h||₂（h 系数和为 1 的
        # box 族不例外）；gust 的 E|lp| = 0.798·||h||₂
        self._norm_car = float(h_car.norm()) or 1.0
        self._norm_band = float(h_band.norm()) or 1.0
        self._norm_lp1 = float(h_lp1.norm()) or 1.0
        self._norm_lp2 = float(h_lp2.norm()) or 1.0
        self._inv_lp1 = self._INV_SQRT2PI * self._norm_lp1  # f = |lp| / (0.798·σ)
        self._inv_lp2 = self._INV_SQRT2PI * self._norm_lp2

        # 上下文缓存长度（各域取最长核）
        self._a_ctx = drive_k - 1                    # audio 域：drive(最长)+prox
        self._car_ctx = h_car.shape[2] - 1           # 载波/带通共用同一噪声输入
        self._flick_ctx = h_lp2.shape[2] - 1         # gust 两尺度共用同一噪声输入
        self._noise_ctx = self._k_floor.shape[2] - 1  # 三角核长 2k-1
        self._max_block = int(self._MAX_BLOCK_SEC * sr)
        self._gpu = None
        self.reset()

    def reset(self):
        """重置所有有状态的缓存（warmup 后调用，确保离线从干净状态开始）。"""
        self._prev_audio = None
        self._prev_car = None
        self._prev_flick = None
        self._prev_noise = None
        self._rngs = [None, None, None]
        self._rng_device = None
        self._pools = [None, None, None]
        self._pool_pos = [0, 0, 0]

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
    def _causal_conv1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """因果1D卷积：只左侧pad(k-1)，输出长度=输入长度。"""
        k = kernel.shape[2]
        x_padded = F.pad(x, (k - 1, 0))
        return F.conv1d(x_padded, kernel)

    def _tail(self, x_full: torch.Tensor, n: int) -> torch.Tensor:
        """上下文卷积输出取当前块的尾部。"""
        return x_full[0, 0, x_full.shape[2] - n:]

    def process(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        # 旁路：强度 0 时不做任何处理，也不推进块间缓存。
        if intensity <= 0:
            return audio
        n_total = audio.shape[0]
        if n_total > self._max_block:
            # 整段离线调用：内部按 ≤1s 子块处理（上下文每子块独立衔接），
            # 保证与流式分块逐样本一致
            outs = [self._process_block(audio[i:i + self._max_block], intensity)
                    for i in range(0, n_total, self._max_block)]
            return torch.cat(outs)
        return self._process_block(audio, intensity)

    def _kernels_on(self, device: torch.device):
        """核的 GPU 缓存（首次/换设备时迁移）。"""
        if self._gpu is None or self._gpu[1] != device:
            self._gpu = (tuple(t.to(device) for t in (
                self._k_drive, self._k_prox, self._k_floor,
                self._k_car, self._k_band, self._k_lp1, self._k_lp2,
            )), device)
        return self._gpu[0]

    def _process_block(self, audio: torch.Tensor, intensity: float) -> torch.Tensor:
        intensity = intensity * self.INTENSITY_SCALE
        n = audio.shape[0]
        device = audio.device
        k_drive, k_prox, k_floor, k_car, k_band, k_lp1, k_lp2 = self._kernels_on(device)

        # ═══════════════════════════════════════════════════════════
        # 1. audio 域：语音慢包络（气流驱动）+ 近讲低通（共用上下文）
        # ═══════════════════════════════════════════════════════════
        if self._prev_audio is not None:
            audio_ctx = torch.cat([self._prev_audio, audio])
        else:
            audio_ctx = audio
        self._prev_audio = audio_ctx[-self._a_ctx:].clone()
        a3 = audio_ctx.view(1, 1, -1).abs()
        env = self._tail(self._causal_conv1d(a3, k_drive), n)
        prox = self._tail(self._causal_conv1d(audio_ctx.view(1, 1, -1), k_prox), n)

        # ═══════════════════════════════════════════════════════════
        # 2. 载波 + 带通（噪声 A，级联 box 合并 FIR，RMS 归一）
        # ═══════════════════════════════════════════════════════════
        noise_a = self._noise(0, n, device)
        if self._prev_car is not None:
            car_ctx = torch.cat([self._prev_car, noise_a])
        else:
            car_ctx = noise_a
        self._prev_car = car_ctx[-self._car_ctx:].clone()
        c3 = car_ctx.view(1, 1, -1)
        car = self._tail(self._causal_conv1d(c3, k_car), n) / self._norm_car
        band = self._tail(self._causal_conv1d(c3, k_band), n) / self._norm_band

        # ═══════════════════════════════════════════════════════════
        # 3. 阵性 gust（噪声 B，两尺度低通噪声的 |·| 均值归一后相乘，
        #    软上限防偶发爆点）
        # ═══════════════════════════════════════════════════════════
        noise_b = self._noise(1, n, device)
        if self._prev_flick is not None:
            flick_ctx = torch.cat([self._prev_flick, noise_b])
        else:
            flick_ctx = noise_b
        self._prev_flick = flick_ctx[-self._flick_ctx:].clone()
        f3 = flick_ctx.view(1, 1, -1)
        l1 = self._tail(self._causal_conv1d(f3, k_lp1), n) / self._inv_lp1
        l2 = self._tail(self._causal_conv1d(f3, k_lp2), n) / self._inv_lp2
        gust = self.GUST_CEIL * torch.tanh(l1 * l2 / self.GUST_CEIL)

        # ═══════════════════════════════════════════════════════════
        # 4. 气流驱动：连续耦合语音包络（无离散触发，电平永远连续）
        # ═══════════════════════════════════════════════════════════
        sp = (env / self.DRIVE_REF).clamp(0.0, self.DRIVE_MAX).pow(self.DRIVE_EXP)
        drive = self.DRIVE_BASE + self.DRIVE_SPEECH * sp
        breath = gust * drive * (car + band * (drive * self.BAND_GAIN)) \
            * (self.BREATH_GAIN * intensity)

        # ═══════════════════════════════════════════════════════════
        # 5. 稳态底噪（噪声 C）
        # ═══════════════════════════════════════════════════════════
        noise_c = self._noise(2, n, device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise_c])
        else:
            noise_ctx = noise_c
        self._prev_noise = noise_ctx[-self._noise_ctx:].clone()
        floor = self._tail(self._causal_conv1d(noise_ctx.view(1, 1, -1), k_floor), n) \
            * self.FLOOR_GAIN * intensity

        voice = audio + prox * self.PROXIMITY_GAIN * intensity
        return voice + breath + floor


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

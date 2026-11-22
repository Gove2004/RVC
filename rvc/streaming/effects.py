"""音频效果器 — RMS 混合 / SOLA / 麦克风气流声，统一 torch.Tensor（GPU）接口。

输出侧处理链：
  RMS 混合 → SOLA 拼接 → 麦克风气流声 → 硬件输出

所有效果器在 setup() 时初始化，热路径纯向量化（零 Python 循环）。
"""
import numpy as np
import torch
import torch.nn.functional as F

from rvc.dsp.rms import apply_rms_mix
from rvc.dsp.sola import apply_sola

# 气流感噪声的固定种子：离线渲染逐样本可复现（实时链路对噪声序列无要求）
AIRFLOW_NOISE_SEED = 0x5EED_A17


class RmsMixEffect:
    """RMS 响度包络混合 — 让转换后音量参考原始音量。"""

    def __init__(self):
        self._samples_per_centis = 0

    def setup(self, sr: int):
        self._samples_per_centis = sr // 100

    def process(self, infer: torch.Tensor, ref: torch.Tensor, rms_mix: float) -> torch.Tensor:
        if rms_mix >= 1.0:
            return infer
        return apply_rms_mix(ref, infer, rms_mix, self._samples_per_centis)


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

    def setup(self, block_samples: int, sola_buffer_samples: int,
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
    """麦克风气流声 — 低频 rumble 质感 + 物理时序吸气节奏。

    质感层
    ------
    100Hz rumble（呼呼声主体：吸气事件 + 说话中的气息）+ 85Hz 底噪（只铺静音段）+
    近讲低频搁架 + 静态软饱和压峰。
    已验证无效、不要再加的层：4kHz 辅音嘶声与 250Hz–1kHz 吸气湍流带（宽带/带通
    白噪声无论怎么门控听感都是「滋/沙」，不是气声）、75Hz resonance（与 100Hz
    主瓣重叠，单听不可辨）、AM 幅度调制（人声幅度变化低于听阈）、任何叠加在语音
    旁边的噪声层（平谱乘包络 / 带通乘人声 / 恒定底噪 → 全部被判「沙沙」）、
    空气吸收/话筒频响（切高频 = 「把人声的高频都切了」）。

    节奏层：物理时序 + 气量，零事件触发、零随机
    -------------------------------------------
    静音持续 ≥0.15s 才开始吸气（词间闭止不误吸），起升速率用 smoothstep 在 0.10s
    内渐入（env 以零斜率离开 0——硬切换的拐点听感是停顿后「呼」地冒出），之后
    τ=0.12s 升起、0.40s 到顶、τ=0.22s 自行回落；说话恢复再 τ=0.05s 快切兜底。
    ≥0.4s 的停顿里整条弧在停顿内部走完，更短的停顿被快切截断——那里电平也最低。

    气量（事件深度）：说话量慢积分 τ=0.8s，静音开始处锁存成本次 depth，开口时
    按 0.5 结算偿还——短句只换到浅吸，长句后才有深吸，节奏越快吸气越轻。depth
    在停顿中恒定，弧不被自身历史调制。

    说话中的气息（air）：同一个 rumble 乘 τ=0.25s 慢包络（按 `AIR_REF` 归一并限幅
    [0,1]），跟着整句的力度起伏、随语句衰减，所以读成「人声里的气」，而不是独立
    噪声床。rumble 不随语音门控退让——退让会全程留一层恒定隆隆（「说话中那层持续
    底噪不对」），而说话段完全没有又会判「说话时候没声音」。
    吸气弧必须在停顿内升落完毕：只升不落、到开口才到顶的形态会把呼呼压在停顿后的
    第一个字上再被硬切，位置感和断句都突兀。
    活跃度门控（τ=1s EMA）保证长静音后不再呼吸、冷启动静音不凭空吸气。

    块间一致性（硬规则）
    -------------------
    - 缓存滤波**输入**而非输出（_prev_audio / _prev_mod / _prev_noise /
      _prev_gate / _prev_audio_prox）
    - 因果卷积只左侧 pad(k-1)；F.conv1d 是互相关，gate EMA 核已 flip
    - 噪声**每消费者独立流**（调制噪声 / 风噪各一个生成器 + 固定 16384 补给池
      顺序消费）：单池多消费者的「值→消费者」映射随块长变化，逐块 randn 的值序
      也与调用长度相关，两者都会让整段渲染 ≠ 分块渲染
    - dur / env / act / fuel / held 五条仿射递推（e'=a·e+b）用并行前缀扫描精确实现
      （组合律 (a_r,b_r)∘(a_l,b_l)=(a_l·a_r, a_r·b_l+b_r)），跨块用 GPU 标量尾
      衔接。递推本身与分块无关；整段与分块渲染只在浮点舍入意义下一致（conv1d
      的结合序随输入长度变化，差异 ulp 量级，见 tests/test_sola_effects.py）
    - 软饱和 drive 逐样本跟随吸气包络：改用块内均值会让均值随块长变，整段与
      分块渲染不一致
    - 零 .item() / Python 真值 / 0-dim 张量索引（GPU 标量只经张量算术）
    """

    # ── 增益（不依赖sr）──
    BREATH_GAIN = 4.6          # rumble 增益：滑条 50% 停顿峰值 ≈ -13..-19dB（随气量）
    AIR_GAIN = 0.9             # 说话中的气息：同一 rumble 乘慢包络（跟着整句走）
    AIR_REF = 0.08             # 气息满幅的语音包络（≈中等音量元音）
    AIR_TAU = 0.25             # 气息包络平滑：音节级 AM 会把噪声听成「一吐一吐」
    FLOOR_GAIN = 0.04          # 底噪增益（只铺静音段，说话中不铺床）
    INTENSITY_GAMMA = 1.8      # GUI 滑块 → 内部强度：s^1.8（100%=设计满幅，50%≈29%，70%≈53%）

    # ── 近讲效应（proximity effect）──
    PROXIMITY_GAIN = 0.4       # 低频提升量（约 +3dB @ 拉满）

    # ── 软饱和（压峰，模拟麦克风电路余量）──
    SATURATION_BASE = 0.8      # 静态 drive：恒开（不乘 intensity），无吸气时也轻微压峰
    SATURATION_AIR = 1.5       # 吸气事件期间的额外 drive（乘 intensity，只落在静音段）

    # ── 语音门控（单源定义）──
    GATE_MID = 0.08            # sigmoid 中点（RMS 包络量纲）
    GATE_STEEP = 30.0          # sigmoid 陡度
    GATE_EXP = 0.5             # 非线性门控指数（中等音量更明显）
    GATE_EMA_FIRST = 0.08      # 门控 EMA 平滑权重
    GATE_EMA_DECAY = 0.92

    # ── 吸气包络（节奏层）：物理时序，无事件触发 ──
    SILENCE_FLOOR = 0.015      # 包络低于此 = 静音
    INHALE_MIN_PAUSE = 0.15    # 静音不足此时长不吸气（词间闭止不误吸）
    INHALE_RATE_RAMP = 0.10    # 起升速率 smoothstep 渐入宽度（零斜率起弧）
    INHALE_RISE_TAU = 0.12     # 吸气上升时间常数
    INHALE_PEAK_S = 0.40       # 静音到此时长吸气到顶，转入回落：整条弧在停顿内走完
    INHALE_FALL_TAU = 0.22     # 到顶后的回落（开口时气息已在衰减，不会被硬切）
    INHALE_RELEASE_TAU = 0.05  # 说话恢复时吸气快速切断

    # ── 气量（事件深度）：说话量慢积分；一次吸气在开口时按比例结算 ──
    #    短句 → 浅吸；长句 → 一次深吸；连续换气时后一次自然变浅。
    #    depth 在静音开始处锁存（停顿中不再变化，弧不被自身历史调制）。
    FUEL_TAU = 0.8             # 气量积累时间常数（0.5s 短句即可积到 ~0.45）
    FUEL_REF = 0.05            # 气量归一参考包络（≈刚可闻语音）
    FUEL_BREATH_USE = 0.5      # 一次吸气消耗的气量比例（开口时结算）

    # ── 说话活跃度门控：长静音后不再呼吸、冷启动不凭空吸气 ──
    ACT_TAU = 1.0              # 活跃度 EMA 时间常数
    ACT_THRESHOLD = 0.05       # 活跃度归一阈值（≈刚可闻语音的包络）

    _MAX_BLOCK_SEC = 1.0       # 内部子块上限（整段离线调用时切分；流式块远小于此）
    _NOISE_CHUNK = 16384       # 噪声池补给单位（CUDA randn 值序与调用块长相关）
    # 跨块仿射递推的槽位数（dur/env/act/fuel/held/air 六条）。
    # 批扫描缓冲与 tail 缓冲都按它一次性分配，热路径不再增长。
    _SCAN_SLOTS = 6
    # 槽位编号。**必须与 _process_block 里的批分组一一对应且同批连续**：
    # _scan_group 用 tail[slot : slot+k] 连续切片取本批初值，写回时也按同一
    # 顺序。槽位与批内顺序错位会让某条递推读到**别人**的 tail——曾把 air 排在
    # 末位、批内却是第 3 条，导致 act 每块都读到同一份旧值、EMA 完全不累积
    # （分块渲染漂移 0.138，整段仍正常，故只在流式链路暴露）。
    # 分批：批1={dur,act,air} 批2={env,fuel} 批3={held}
    _S_DUR, _S_ACT, _S_AIR = 0, 1, 2
    _S_ENV, _S_FUEL = 3, 4
    _S_HELD = 5

    # @48k 参考核大小（setup中按sr比例缩放）
    _REF_SR = 48000
    _REF_SLOW_K = 2400         # 50ms box低通（包络/调制）
    _REF_FLOOR_K = 180         # 85Hz三角核
    _REF_BREATH_K = 150        # 100Hz三角核
    _REF_PROX_K = 51           # box 低通 -3dB ≈417Hz（近讲效应）
    _REF_GATE_LEN = 40         # EMA 窗口 ≈0.83ms（τ≈0.25ms）

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
        self._fall_a = 1.0
        self._rel_a = 1.0
        self._alpha_act = 0.0      # 活跃度 EMA 步进
        self._act_a = 1.0
        # 上下文缓存长度
        self._slow_ctx = 0
        self._combo_ctx = 0
        self._gate_ctx = 0
        self._prox_ctx = 0
        self._max_block = 0
        # 块间上下文缓存（存滤波输入而非输出）
        self._prev_audio = None       # 音频（slow 包络 2ch 批量卷积共用）
        self._prev_mod = None         # 调制噪声输入
        self._prev_noise = None       # 风噪输入（combo 2ch 共用）
        self._prev_gate = None        # 门控 EMA 的输入
        self._prev_audio_prox = None  # 近讲低通的输入
        # GPU 标量状态（跨块衔接；只经张量算术更新，绝不 .item()）。
        # 六条递推的末状态统一放在一个预分配 (6,1) 缓冲里，按 _S_* 槽位索引——
        # 分开存会在每次批扫描时 cat 组装 tail，正是要消除的那类每块分配。
        self._scan_tail = None        # (6,1) dur/env/act/fuel/held/air 的块尾
        self._dur_prev_buf = None     # (1,) 上一块 dur 末值的副本（见 _process_block）
        # 复用缓冲（热路径零分配）
        self._batch_in = None       # (2,1,ctx_n)：调制噪声/包络
        self._scan_ab = None        # (6,2,n)：批扫描的 a/b 输入
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
        prox_k, gate_len = k(self._REF_PROX_K), max(2, k(self._REF_GATE_LEN))

        # slow：box 低通（包络 + 调制噪声）
        self._slow_kernel = self._box(slow_k)
        self._prox_kernel = self._box(prox_k)
        # combo：分组卷积组合核（2 通道：breath/floor；resonance 已删——
        # 75Hz 与 100Hz 主瓣重叠，仅占效果 5.1%）
        combo_size = 2 * max(floor_k, breath_k) - 1
        combo = torch.zeros(2, 1, combo_size)
        for i, (kern, ref_k) in enumerate((
                (self._triangular_kernel(breath_k), self._REF_BREATH_K),
                (self._triangular_kernel(floor_k), self._REF_FLOOR_K))):
            # 三角核的白噪声增益 ∝ 1/sqrt(k)：按参考网格归一，电平才与采样率无关
            combo[i, 0, :kern.shape[2]] = kern[0, 0] * (
                kern.shape[2] / (2 * ref_k - 1)) ** 0.5
        self._combo_kernel = combo
        # gate：指数 EMA（反转使最大权重对应最新样本）。有限窗截断后 DC 增益
        # ≈0.964（0.08·Σ0.92^k, k<40），门控开到最大时底噪仍有 ≈3.6% 泄漏
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
        self._fall_a = float(np.exp(-1.0 / (self.INHALE_FALL_TAU * sr)))
        self._rel_a = float(np.exp(-1.0 / (self.INHALE_RELEASE_TAU * sr)))
        self._alpha_act = 1.0 - float(np.exp(-1.0 / (self.ACT_TAU * sr)))
        self._act_a = 1.0 - self._alpha_act
        self._fuel_a = float(np.exp(-1.0 / (self.FUEL_TAU * sr)))
        self._fuel_alpha = 1.0 - self._fuel_a
        self._air_a = float(np.exp(-1.0 / (self.AIR_TAU * sr)))
        self._air_alpha = 1.0 - self._air_a
        self._gpu = None
        self.reset()

    def reset(self):
        """重置所有有状态的缓存（warmup 后调用，确保离线从干净状态开始）。"""
        self._prev_audio = None
        self._prev_mod = None
        self._prev_noise = None
        self._prev_gate = None
        self._prev_audio_prox = None
        # 六条递推的块尾统一归零（None = 流起始，与旧语义一致）
        self._scan_tail = None
        self._dur_prev_buf = None
        self._rngs = [None, None]
        self._rng_device = None
        self._pools = [None, None]
        self._pool_pos = [0, 0]

    @property
    def _env_tail(self):
        """吸气包络（env）递推的块尾值 —— 只读观测点。

        六条递推的末状态已合并进 _scan_tail（见 _SCAN_SLOTS 的说明），本属性
        只是它的 env 行视图，供测试与调试读取，不参与写入。
        """
        if self._scan_tail is None:
            return None
        return self._scan_tail[self._S_ENV, 0]

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
        递推本身精确（与逐样本循环在实数意义下一致），但结合序随序列长度变化，
        所以只在浮点舍入意义下与分块无关——漂移 ulp 量级，远小于可闻阈值。

        **支持批处理**：a/b 为 (K, n) 时沿 dim=1 扫描，tail 形状 (K, 1)。
        多条互不依赖的递推合并成一次调用能把 cat 次数按 K 倍摊薄——本函数是
        热路径上唯一的 O(n log n)·cat 开销来源（单块 5~6 处调用共 156 次 cat，
        占全部张量分配的 96%），批处理后 GPU 侧耗时降约 5 倍。
        数值与逐条调用**逐位相同**（每条的递推式与结合序都未变）。
        """
        batched = a.dim() == 2
        n = a.shape[1] if batched else a.shape[0]
        A, B = a.clone(), b.clone()
        step = 1
        while step < n:
            if batched:
                a_r, b_r = A[:, step:], B[:, step:]
                a_l, b_l = A[:, :-step], B[:, :-step]
                # 先算 B（要用的旧 A），再更新 A
                B = torch.cat([B[:, :step], a_r * b_l + b_r], dim=1)
                A = torch.cat([A[:, :step], a_l * a_r], dim=1)
            else:
                a_r, b_r = A[step:], B[step:]
                a_l, b_l = A[:-step], B[:-step]
                # 先算 B（要用的旧 A），再更新 A
                B = torch.cat([B[:step], a_r * b_l + b_r])
                A = torch.cat([A[:step], a_l * a_r])
            step <<= 1
        if tail is not None and tail.device == a.device:
            return A * tail + B
        return B

    def _scan_group(self, k: int, a_parts, b_parts, tail_slot: int, n: int):
        """把 k 条互不依赖的仿射递推合并成一次批处理扫描。

        a_parts / b_parts：长度 k 的「如何构造第 i 条系数序列」的产物列表，
        每项形状 (n,)。tail 从预分配的 self._scan_tail 取第 tail_slot 行。

        复用两块缓冲：
        - self._scan_ab  (k, 2, n)：拼 a/b 的批输入（避免 stack 每块分配）
        - self._scan_tail (S, 1)   ：跨块标量尾（原地写，不 cat）

        返回 (k, n) 结果；调用方按行取用。写回 tail 由调用方负责
        （见 _scan_store_tail），以保持「哪一行对应哪条递推」的显式性。
        """
        device = a_parts[0].device
        ab = self._scan_ab
        if ab is None or ab.shape[0] < k or ab.shape[2] != n:
            ab = torch.empty(max(k, self._SCAN_SLOTS), 2, n, device=device)
            self._scan_ab = ab
        for i in range(k):
            ab[i, 0].copy_(a_parts[i])
            ab[i, 1].copy_(b_parts[i])
        tail = self._scan_tail
        if tail is None:
            tail = torch.zeros(self._SCAN_SLOTS, 1, device=device)
            self._scan_tail = tail
        return self._scan_affine(ab[:k, 0], ab[:k, 1], tail[tail_slot:tail_slot + k])

    def _scan_store_tail(self, result: torch.Tensor, slots) -> None:
        """把批扫描结果的末列写回预分配 tail 缓冲的指定行（原地，不分配）。

        slots：本批各条递推在 self._scan_tail 中的行号。
        用 detach 视图写入：tail 是跨块状态，不得带进 autograd 图。
        """
        tail = self._scan_tail
        for row, slot in enumerate(slots):
            tail[slot, 0] = result[row, -1].detach()

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
        # 幂曲线：滑块 100% 才是设计满幅；线性映射会把上限压到听不见，
        # 且中低段手感偏冲（1.8 次幂下 50%≈29%、70%≈53%）
        intensity = intensity ** self.INTENSITY_GAMMA
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
        # 3. 吸气包络（节奏层）：物理时序而非事件触发——
        #    静音里升起（≥0.15s 才起、τ=0.12s），到 0.40s 到顶后自行回落
        #    （τ=0.22s）→ 整条吸气弧在停顿内部走完，开口时气息已经在衰减，
        #    既不会压住第一个字、也不会被切断成「咔」；说话恢复仍保留
        #    τ=0.05s 快切作为短停顿（弧没走完）的兜底。
        #    活跃度门控在 4b（需语音包络）。
        #    dur / env / act 三条仿射递推并行前缀扫描精确实现，
        #    跨块 GPU 标量尾衔接——与分块无关。
        # ═══════════════════════════════════════════════════════════
        sil_ind = (envelope < self.SILENCE_FLOOR).float()

        # ── 第 1 批扫描：{dur, act, air} —— 三条互不依赖，合并为一次调用 ──
        # dur 只吃 sil_ind；act/air 只吃 envelope（均已就绪）。合并后 cat 次数
        # 从 3×2×log2(n) 降到 1×2×log2(n)，数值逐位相同（见 _scan_affine）。
        # 3a. 静音持续时长（样本）：dur' = sil ? dur+1 : 0
        #     ⇒ 逐步系数 a=sil, b=sil（sil=0 时归零，sil=1 时线性累加）
        #
        # dur_prev_tail 必须在批扫描**之前**取走：它下面要用来算 resume
        # （本块首样本的「静音→有声」跃变判定，需上一块的末值），而
        # _scan_store_tail 会把本块新值原地写回同一个缓冲。若直接引用视图，
        # 读到的是本块自己的末值，resume 整块错位（实测输出偏差 6e-3 且跨块累积）。
        # 用预分配的 1 元素缓冲承接旧值，避免每块 clone 一次分配。
        if self._scan_tail is None:
            dur_prev_tail = None
        else:
            if self._dur_prev_buf is None:
                self._dur_prev_buf = torch.zeros(1, device=device)
            self._dur_prev_buf.copy_(self._scan_tail[self._S_DUR])
            dur_prev_tail = self._dur_prev_buf
        voiced_n = (envelope * (1.0 / self.FUEL_REF)).clamp(0.0, 1.0)
        b_act = envelope * self._alpha_act
        b_air = envelope * self._air_alpha
        a_const_act = torch.full((n,), self._act_a, device=device)
        a_const_air = torch.full((n,), self._air_a, device=device)
        batch1 = self._scan_group(
            3,
            (sil_ind, a_const_act, a_const_air),
            (sil_ind, b_act, b_air),
            self._S_DUR, n,
        )
        dur = batch1[0]
        act = batch1[1]
        air_env = batch1[2]
        self._scan_store_tail(batch1, (self._S_DUR, self._S_ACT, self._S_AIR))

        # 3b. 吸气包络：分段仿射系数（未满最短停顿持零 / 起升 / 到顶后回落 /
        #     有声快速切断）。起升速率 k 按 smoothstep 从 0 渐入：硬切换会让
        #     env' 在起弧瞬间从 0 跳到 α，而一阶弧的斜率最大值恰在起点，包络
        #     因此带一个拐点，听感是停顿后「呼」地冒出。渐入后 env 以零斜率
        #     离开 0（前 50ms 只到峰值的 ~10%），峰值时刻不变。
        dur_s = dur * (1.0 / self._sr)
        late = (dur_s > self.INHALE_PEAK_S).float()
        u = ((dur_s - self.INHALE_MIN_PAUSE) / self.INHALE_RATE_RAMP).clamp(0.0, 1.0)
        k = u * u * (3.0 - 2.0 * u) * sil_ind * (1.0 - late)
        a_env = ((1.0 - sil_ind) * self._rel_a
                 + sil_ind * (1.0 - late) * (1.0 - k * self._alpha_rise)
                 + sil_ind * late * self._fall_a)
        b_env = k * self._alpha_rise
        # 开口（静音→有声）的那一个样本结算上一次吸气：气量按比例打掉，
        # 之后由说话量重新积累——短句只换到浅吸，连续换气逐次变浅。
        if dur_prev_tail is None:
            dur_prev = torch.cat([torch.zeros(1, device=device), dur[:-1]])
        else:
            dur_prev = torch.cat([dur_prev_tail.reshape(1), dur[:-1]])
        resume = ((dur == 0) & (dur_prev > 0)).float()
        a_fuel = (torch.full((n,), self._fuel_a, device=device)
                  + resume * ((1.0 - self.FUEL_BREATH_USE) - self._fuel_a))

        # ── 第 2 批扫描：{env, fuel} —— 都只依赖 dur，合并为一次调用 ──
        batch2 = self._scan_group(
            2,
            (a_env, a_fuel),
            (b_env, voiced_n * self._fuel_alpha),
            self._S_ENV, n,
        )
        env_in = batch2[0]
        fuel = batch2[1]
        self._scan_store_tail(batch2, (self._S_ENV, self._S_FUEL))

        # ── 第 3 批扫描：{held} —— 依赖 fuel，只能单独一批 ──
        # 静音开始处锁存气量：整段停顿用同一个 depth，弧不被自身历史调制
        batch3 = self._scan_group(
            1,
            (sil_ind,),
            ((1.0 - sil_ind) * fuel,),
            self._S_HELD, n,
        )
        held = batch3[0]
        self._scan_store_tail(batch3, (self._S_HELD,))
        depth = held.clamp(0.0, 1.0)

        # ═══════════════════════════════════════════════════════════
        # 4. 门控：sigmoid → 非线性 → EMA平滑（不做辅音增强——说话中
        #    rumble 随辅音突跳是「隆隆起伏感」的来源）
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
        # 4b. 活跃度门控 / 4c. 气量 / 4d. 气息包络：三条递推已在上面的
        #     第 1、2 批扫描中一并算完（act / fuel / held / air_env），
        #     此处只做归一化与门控组合。
        # ═══════════════════════════════════════════════════════════
        act_n = (act / self.ACT_THRESHOLD).clamp(0.0, 1.0)

        inhale = env_in * act_n * depth

        # ═══════════════════════════════════════════════════════════
        # 5. 风噪声（combo 分组卷积，2 通道）
        # ═══════════════════════════════════════════════════════════
        noise = self._noise(1, n, device)
        if self._prev_noise is not None:
            noise_ctx = torch.cat([self._prev_noise, noise])
        else:
            noise_ctx = noise
        self._prev_noise = noise_ctx[-self._combo_ctx:].clone()

        noise_2ch = noise_ctx.view(1, 1, -1).expand(1, 2, -1)
        combo_out = self._causal_conv1d(noise_2ch, k_combo, groups=2)
        noise_n = noise_ctx.shape[0]
        breath_low = combo_out[0, 0, noise_n - n:noise_n]
        floor_low = combo_out[0, 1, noise_n - n:noise_n]

        # wind 缓慢起伏（调制噪声自调制）
        wind_mod = 1.0 + 0.4 * (mod_low - 1.0) / 0.15

        # 呼呼声：吸气事件（静音段）
        rumble = breath_low * inhale * wind_mod * self.BREATH_GAIN
        breath = rumble * intensity

        # 说话中的气息：同一 rumble 乘 τ=0.25s 慢包络——跟着整句的力度走，
        # 不跟音节（噪声随音节调幅会被听成「一吐一吐」），句子停下后随包络消失。
        # air_env 已在第 1 批扫描中算好（只依赖 envelope）。
        air_amt = (air_env * (1.0 / self.AIR_REF)).clamp(0.0, 1.0)
        air = breath_low * air_amt * wind_mod * self.AIR_GAIN * intensity

        # 底噪只铺静音段（语音门控越强越低）：说话中只剩 EMA 截断泄漏的
        # ≈3.6% 底（见 setup 的 gate 核注释），停顿里是完整的「麦克风开着」底
        floor = floor_low * (1.0 - envelope_gate) * intensity * self.FLOOR_GAIN

        # ═══════════════════════════════════════════════════════════
        # 6. 近讲效应低频搁架（box 低通 -3dB ≈417Hz 以下提升）
        # ═══════════════════════════════════════════════════════════
        if self._prev_audio_prox is not None:
            audio_prox_ctx = torch.cat([self._prev_audio_prox, audio])
        else:
            audio_prox_ctx = audio
        self._prev_audio_prox = audio_prox_ctx[-self._prox_ctx:].clone()
        audio_low = self._causal_conv1d(
            audio_prox_ctx.view(1, 1, -1), k_prox
        )[0, 0, audio_prox_ctx.shape[0] - n:]
        audio_prox = audio + audio_low * self.PROXIMITY_GAIN * intensity

        # ═══════════════════════════════════════════════════════════
        # 7. 软饱和（x / (1 + |x| · drive)）：只压峰，小信号增益为 1。
        #    drive 只跟随吸气事件，**不跟随语音门控**——后者会随句读起伏
        #    改变压缩量，实测把说话段的电平差拉到 18dB 级泵浦（音节抽气感），
        #    断句处最刺眼。吸气发生在静音段，那里没有语音可压，所以这条
        #    耦合对呼呼质感没有代价。
        #    ⚠️ drive 必须逐样本：用块内均值会让块长改变结果（实测 max|Δ| 2.7e-2）
        #    静态项有意不乘 intensity：它是「麦克风电路余量」的固定压峰，滑条只管
        #    气流层；乘上去会让 0% 变成完全无压峰、人声峰形态随滑条变化。
        # ═══════════════════════════════════════════════════════════
        drive = self.SATURATION_BASE + inhale * (self.SATURATION_AIR * intensity)
        audio_sat = audio_prox / (1.0 + torch.abs(audio_prox) * drive)

        # ═══════════════════════════════════════════════════════════
        # 8. 混合输出：人声（近讲+饱和）+ 呼呼声 + 底噪
        # ═══════════════════════════════════════════════════════════
        return audio_sat + breath + air + floor


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
        self.sola.setup(block_samples, sola_buffer_samples, sola_search_samples, device)
        self.airflow.setup(sr, device)

    def reset(self):
        """重置所有有状态的效果器（warmup 后调用）。"""
        self.sola.reset()
        self.airflow.reset()

    def process_output(self, infer, ref, rms_mix, airflow=0.0):
        """输出侧处理：RMS 混合 → SOLA → 麦克风气流声。"""
        infer = self.rms_mix.process(infer, ref, rms_mix)
        infer = self.sola.process(infer)
        # 量程钳制放在消费点：配置文件可被外部改写，滑块量程不构成保证
        return self.airflow.process(infer, min(max(airflow, 0.0), 1.0))

"""从参考输入块向转换输出块「搬运」真实感。

转换模型的输出天然过于干净：训练数据的平均化把底噪、齿音、近讲低频、话筒
距离感一起抹平了，听感就是「像合成器」。凭空合成噪声解决不了——噪声与语音
内容不相关，耳朵立刻能分辨出「加了噪声」而不是「录音」。参考输入块里本来就
带着真实录音链路的痕迹，本模块只做**搬运**：用什么电平、什么频带、什么谱形
全部由输入的统计量决定，纹理与内容天然同步。

搬运的边界：**只搬「痕迹」，不搬「身份」**。直接混入输入信号会把原声按比例
漏进输出（residual 早期版本实测与原声余弦 0.75、安静段相对电平仅 −1.4dB，
就是可闻的双重声）。所以含音色成分的那一级必须去掉谐波结构——幅度包络取自
输入的底噪谱，相位换成随机值，音色身份在结构上不可能残留。

四级各治一种「不真实」，互相不可替代：
- residual：非谐波底噪。底噪是录音链路的指纹，缺了它静音段就是「死静」，
  谐波再准也不像人声。取输入的底噪幅度包络（频率方向中值滤掉谐波峰）+ 随机
  相位重建，按「噪声样程度」定增益。
- breath：>4kHz 气息与齿音。转换模型的高频常被过度平滑，/s/ /f/ 发闷、字头
  含糊；这一带几乎不含音色身份，直接搬输入的高频带最保真。
- rumble：近讲低频搁架 + tanh 软饱和。低频隆起是人耳判断话筒距离的主线索，
  但纯提升会让爆破音低频过冲削顶，所以用 tanh 兜住峰值。
- distance：慢速随机游走（OU 过程）的电平 + 谱倾斜。真人对着手机说话时距离
  一直在微动；电平完全恒定反而像「贴脸的合成器」。

所有强度默认 0 = 逐位旁路：任何一级开大都会从「更真实」变成「加特效」，所以
默认值必须安全。热路径约束：全向量化（无逐样本 Python 循环、无 `.item()`、无
数据相关形状的布尔索引），跨块状态固定尺寸，不随时间增长。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# RealismParams 定义在 rvc.core.config：分层契约要求 core(0) 不得依赖 dsp(2)，
# 而参数必须活在 config 里才能被 PARAM_GROUPS 的序列化/拷贝自动带上。
from rvc.core.config import RealismParams

__all__ = ["RealismChain", "RealismParams"]

# FIR 抽头数。每块最多 3 次 conv1d；63 抽头在 40kHz/250ms 块上约 0.75M MAC，
# 相对合成器可忽略。抽头再少会让 200Hz 切点漏进中频，近讲检测变钝。
_TAPS = 63
_LOW_CUT_HZ = 200.0     # 近讲低频与中频的分界
_HIGH_CUT_HZ = 4000.0   # 气息/齿音带的下沿
_RUMBLE_MAX_DB = 6.0    # 近讲搁架的最大提升量
_DISTANCE_MAX_DB = 1.5  # 远近起伏的电平量程（±1.5dB）
_TILT_MAX = 0.08        # 远近起伏的谱倾斜量（相对高频成分的比例）
_WALK_STD = 0.30        # 随机游走每块的标准差（配回归系数，稳态 |w| 约 0.3~0.6）
_WALK_PULL = 0.15       # 向 0 的回归系数，防止游走长期贴边
_LEVEL_DECAY = 0.85     # 输出电平跟踪的释放系数（约 1.5s 时间常数）
_LEVEL_MATCH_MAX = 2.0  # 输入→输出电平换算上限，防止参考远大于输出时注入爆音
# 底噪幅度包络的中值滤波窗（频率方向，单位 bin）。必须盖住至少两个谐波周期，
# 中值才会落在谐波间隙而不是谐波峰上：41 bin 在 10Hz/bin 下是 410Hz，覆盖
# F0 ≥ ~205Hz。更低频的男声只会让窗口装下更多周期，中值仍是底噪。
_FLOOR_WIN = 41
_EPS = 1e-8
_DEFAULT_SEED = 20240601


def _design_lowpass(cutoff_hz: float, sr: int, taps: int, device: torch.device) -> torch.Tensor:
    """设计线性相位 FIR 低通（Hamming 窗，DC 增益归一化为 1）。

    用 FIR 而非 IIR：IIR 有反馈，跨块必须逐样本递归，无法向量化，与「回调线程
    里零 Python 循环」的约束冲突。FIR 只需保存 taps-1 个历史样本（overlap-save），
    一次 conv1d 出整块，块边界严格连续。cutoff 自动钳到 0.45*sr 内以免越过 Nyquist。
    """
    cutoff = min(float(cutoff_hz), 0.45 * sr)
    m = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2.0
    h = 2.0 * cutoff / sr * torch.sinc(2.0 * cutoff / sr * m)
    h = h * torch.hamming_window(taps, periodic=False, dtype=torch.float64)
    return (h / h.sum()).to(dtype=torch.float32, device=device)


def _highpass_from(lowpass: torch.Tensor) -> torch.Tensor:
    """由低通核取互补高通核（δ - h_lp）。

    必须互补：两核不互补时 low + high ≠ x，中频带留下一条凹口，底噪与气息
    之间出现「空带」，听感是发虚。独立设计两个核几乎不可能互补。
    """
    hp = -lowpass.clone()
    hp[lowpass.shape[0] // 2] += 1.0
    return hp


def _rms(x: torch.Tensor) -> torch.Tensor:
    """块内 RMS，返回 0-dim 张量（不做 `.item()`，避免 GPU→CPU 同步）。"""
    return torch.sqrt(torch.mean(x * x) + _EPS)


def _floor_envelope(mag: torch.Tensor, win: int) -> torch.Tensor:
    """频率方向滑动中值 —— 谐波是窄峰，中值落在谐波间隙的底噪上。

    用中值不用最小值：最小值会挑到相邻谐波之间的谱零点，系统性低估底噪；
    中值忽略少数谐波峰，给出的是该频段的底噪电平。
    """
    pad = win // 2
    padded = F.pad(mag.view(1, 1, -1), (pad, pad), mode="replicate")
    windows = padded.unfold(-1, win, 1)
    return windows.median(dim=-1).values.view(-1)


class RealismChain:
    """有状态的实时真实感链：参考输入块 → 转换输出块的痕迹搬运。

    生命周期：`setup()` 每次开流一次 → 每块 `process()` → 预热/停流后 `reset()`。
    状态只有三处：两个 FIR 的 overlap-save 尾部、块间增益斜坡的起点、距离游走
    与它的 RNG。三者都在 `reset()` 里清零，因此同一 seed 下会话可复现。
    """

    def __init__(self, params: RealismParams | None = None, seed: int = _DEFAULT_SEED):
        self.params = params if params is not None else RealismParams()
        self._seed = int(seed)
        self._taps = 0
        self._n = 0
        self._device = torch.device("cpu")
        self._gen = torch.Generator()
        # 频带滤波器（setup 时按 sr 设计）
        self._lp_low: torch.Tensor | None = None
        self._hp_high: torch.Tensor | None = None
        # 工作缓冲（setup/_alloc 分配，热路径复用，尺寸恒定）
        self._ref_work: torch.Tensor | None = None
        self._out_work: torch.Tensor | None = None
        self._tail_tmp: torch.Tensor | None = None
        self._ramp: torch.Tensor | None = None
        # 跨块状态：4 个增益斜坡起点 + 距离游走 + 输出电平跟踪
        self._gprev = torch.zeros(4)
        self._gprev[2] = 1.0  # distance 的电平增益初值是 1.0（无起伏），不是 0
        self._walk = torch.zeros(())
        self._out_lvl = torch.zeros(())

    # ── 生命周期 ──

    def setup(self, sr: int, block_samples: int, device) -> None:
        """按采样率与块长准备滤波器与工作缓冲（每次开流调用一次）。

        block_samples 与实际块长不一致时，首个 process 会自动重建缓冲（一次
        分配），不影响正确性；稳定后不再重建。
        """
        self._device = torch.device(device)
        self._taps = _TAPS
        self._lp_low = _design_lowpass(_LOW_CUT_HZ, int(sr), _TAPS, self._device)
        self._hp_high = _highpass_from(_design_lowpass(_HIGH_CUT_HZ, int(sr), _TAPS, self._device))
        # 生成器随设备创建：CUDA 与 CPU 的 RNG 实现不同，跨设备不保证同序列
        self._gen = torch.Generator(device=str(self._device))
        self._gprev = torch.zeros(4, device=self._device)
        self._gprev[2] = 1.0
        self._walk = torch.zeros((), device=self._device)
        self._out_lvl = torch.zeros((), device=self._device)
        self._alloc(max(int(block_samples), _TAPS + 1))
        self.reset()

    def reset(self) -> None:
        """清空全部跨块状态（滤波器历史、斜坡起点、游走、RNG）。

        预热跑的是静音块：不清掉会把静音写进 overlap-save 尾部与游走状态，
        正式开流头几十毫秒会「抽」一下。RNG 一并复位，使会话可复现。
        """
        if self._ref_work is not None:
            self._ref_work.zero_()
            self._out_work.zero_()
        self._gprev.zero_()
        self._gprev[2] = 1.0
        self._walk.zero_()
        self._out_lvl.zero_()
        self._gen.manual_seed(self._seed)

    def _alloc(self, n: int) -> None:
        """按块长重建工作缓冲（只在 setup 与块长变化时调用）。

        块长稳定后每块的临时张量尺寸恒定，由缓存分配器复用——回调线程里不
        允许出现随时间增长的分配。
        """
        self._n = n
        pad = _TAPS - 1
        self._ramp = torch.linspace(0.0, 1.0, n, device=self._device)
        self._ref_work = torch.zeros(n + pad, device=self._device)
        self._out_work = torch.zeros(n + pad, device=self._device)
        self._tail_tmp = torch.empty(pad, device=self._device)

    # ── 主入口 ──

    def process(self, out_block: torch.Tensor, ref_block: torch.Tensor) -> torch.Tensor:
        """把参考块的录音痕迹注入输出块，返回同形状、有限、钳在 [-1, 1] 的块。

        返回新张量（入口要做 NaN 消毒，必然是拷贝），调用方直接替换引用即可。

        Args:
            out_block: 转换后的输出块（1-D，work_sr）
            ref_block: 对应的参考输入块（1-D，work_sr；可比输出块长）
        """
        if self._taps == 0:
            # 未 setup：不处理，但仍保证有限（离线路径可能先于 setup 调用）
            return torch.nan_to_num(out_block, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

        p = self.params
        breath, residual = float(p.breath), float(p.residual)
        rumble, distance = float(p.rumble), float(p.distance)

        shape = out_block.shape
        # NaN/Inf 消毒在入口一次做完：之后所有统计量（RMS、频谱中位数、比值）
        # 都是有限值，任何一个 NaN 都不会被除法/乘法放大成整块污染。
        x = torch.nan_to_num(out_block.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
        r = torch.nan_to_num(ref_block.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
        n = x.shape[0]
        if r.shape[0] < n:
            r = F.pad(r, (0, n - r.shape[0]))
        if n != self._n:
            self._alloc(n)

        # 全 0 = 逐位旁路：不碰信号，只留最后一道钳位
        if breath <= 0.0 and residual <= 0.0 and rumble <= 0.0 and distance <= 0.0:
            return x.clamp_(-1.0, 1.0).view(shape)

        # 输入→输出电平换算：注入量要按「痕迹在输入里占多大比例」搬到输出的
        # 绝对电平上。不换算的话输出音量大时气息被淹没、小时则直接爆音。
        # 输出电平走快攻慢放跟踪：静音间隙里当前块 RMS 会掉到 0，若直接用它算
        # 换算系数，间隙里就什么都不注入——而底噪最该出现的位置恰恰是间隙。
        ref_rms = _rms(r)
        self._out_lvl.mul_(_LEVEL_DECAY).clamp_(min=_rms(x))
        lv = torch.clamp(self._out_lvl / ref_rms, max=_LEVEL_MATCH_MAX)

        # 参考块的两个频带。每块只推一次工作缓冲：同一块内推两次会让
        # overlap-save 的尾部状态错位（第二推把第一推的尾部当历史），块边界咔哒。
        # 推送本身必须无条件发生（状态按块推进），但滤波只在用到时算。
        ref_buf = self._push(self._ref_work, r[:n])
        ref_low = self._conv(ref_buf, self._lp_low) if rumble > 0.0 else None
        ref_high = self._conv(ref_buf, self._hp_high) if (breath > 0.0 or rumble > 0.0) else None

        # ── distance：慢速远近起伏 ──
        low_out = None
        if distance > 0.0 or rumble > 0.0:
            low_out = self._conv(self._push(self._out_work, x), self._lp_low)
        if distance > 0.0:
            x, gain = self._stage_distance(x, low_out, distance)
            low_out = low_out * gain  # 搁架要作用在「已受距离调制」的低频上

        # ── rumble：近讲低频搁架 + 软饱和 ──
        if rumble > 0.0:
            # 近讲度量的是**输入**的低频/中频能量比。量输出会出问题：模型本来
            # 就没低频时恒判「不近讲」，于是永远不补低频，正好漏掉要修的情况。
            mid = r[:n] - ref_low - ref_high
            prox = torch.clamp(_rms(ref_low) / (_rms(mid) + _EPS), max=1.0)
            g_low = torch.pow(10.0, rumble * prox * _RUMBLE_MAX_DB / 20.0)
            # 只对低频带做 tanh：中高频保持线性，齿音不会被饱和磨平。线性提升
            # 在爆破音上直接削顶（可闻的「噗」），tanh 把峰值压在 1 以内。
            x = x - low_out + torch.tanh(g_low * low_out)

        # ── breath：>4kHz 气息/齿音回注 ──
        if breath > 0.0:
            # 高频能量占比即天然门限：元音几乎为 0（不注入），/s/ /f/ 时接近 1。
            # 这一级直接搬输入的高频带：>4kHz 几乎不含音色身份（实测与原声余弦
            # 0.003、注入量平坦度 0.216，本质是噪声），无需去谐波。
            hf_ratio = torch.clamp(_rms(ref_high) / ref_rms, max=1.0)
            x = x + ref_high * (lv * self._ramp_gain(0, breath * hf_ratio))

        # ── residual：非谐波底噪回注 ──
        if residual > 0.0:
            bed, noise_like = self._noise_floor_bed(r[:n], n)
            x = x + bed * (lv * self._ramp_gain(1, residual * noise_like))

        return x.clamp_(-1.0, 1.0).view(shape)

    def _noise_floor_bed(self, r: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """用输入的**非谐波底噪谱** + 随机相位重建一条噪声床。

        不能直接搬 `r - 低频`：那是输入 200Hz 以上的整个频谱，谐波与共振峰都在
        里面，等于把原声按比例混进输出（实测注入量与原声余弦 0.75，安静段相对
        本块输出仅 −1.4dB，就是可闻的双重声）。这里只保留底噪的幅度包络、把相位
        换成随机值——电平与谱形仍是输入的统计量，音色身份被抹掉。

        Returns:
            (bed, noise_like)：噪声床，以及「底噪在谱里占多大比例」这个门限。
        """
        mag = torch.fft.rfft(r).abs()
        floor = _floor_envelope(mag, _FLOOR_WIN)
        # 谐波峰稀疏时中位数落在谐波间隙上，比值小；谱平坦（真有底噪）时接近 1
        noise_like = torch.clamp(torch.median(mag) / (torch.mean(mag) + _EPS), max=1.0)
        phase = (torch.rand(floor.shape, generator=self._gen, device=self._device)
                 * 2.0 - 1.0) * math.pi
        # 随机相位不改变幅度谱，按 Parseval 能量与零相位版本相同，所以这里不需要
        # 再除以某个参考 RMS 做归一化——电平已经由 floor 本身定好了。
        return torch.fft.irfft(torch.polar(floor, phase), n=n), noise_like

    def _stage_distance(self, x: torch.Tensor, low_out: torch.Tensor,
                        distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        """慢速远近起伏：OU 随机游走的电平增益 + 谱倾斜，返回 (块, 增益斜坡)。

        必须是有界过程：真人对着手机说话时距离只在几厘米内微动，游走若允许漂移
        就变成「越说越小声」。回归项 + 硬钳双重约束，见 _WALK_PULL。
        """
        step = torch.randn((), generator=self._gen, device=self._device) * _WALK_STD
        # OU 过程：乘性回归 + 加性噪声，再硬钳 [-1, 1]。回归项防止游走长期贴边——
        # 贴边是恒定偏移，听起来像「音量被调小了」，而不是「在动」。
        self._walk.mul_(1.0 - _WALK_PULL).add_(step).clamp_(-1.0, 1.0)
        w = self._walk * distance
        gain = self._ramp_gain(2, torch.pow(10.0, w * (_DISTANCE_MAX_DB / 20.0)))
        tilt = self._ramp_gain(3, w * _TILT_MAX)
        # 倾斜只加在高频成分上：电平归 gain、倾斜归 tilt，两者不重复计权，否则
        # distance 调大时音量会跟着谱倾斜一起漂。
        return gain * (x + tilt * (x - low_out)), gain

    def _ramp_gain(self, idx: int, cur: torch.Tensor) -> torch.Tensor:
        """把某级增益从上一块的值线性过渡到本块的值，返回块内斜坡 [n]。

        直接用块常量增益会在块边界留下台阶：250ms 块上这就是一串 4Hz 的咔哒，
        distance 每块都在变，不斜坡化必然可闻。

        Args:
            idx: 增益槽位（0=breath 1=residual 2=distance 电平 3=distance 倾斜）
            cur: 本块的增益（0-dim 张量）
        """
        prev = self._gprev[idx]
        # 先算完再写回：prev 是 _gprev 的视图，写回后再用就会读到新值
        ramp = prev + (cur - prev) * self._ramp
        self._gprev[idx] = cur.detach()
        return ramp

    # ── FIR 工具（overlap-save） ──

    def _push(self, work: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """把新块推入工作缓冲，返回可直接做 valid 卷积的视图。

        布局：[0, pad) = 上一块的尾部历史，[pad, pad+n) = 本块。因此卷积输出
        长度恰为 n，块边界处滤波器状态连续（不会每块重启出瞬态）。
        """
        pad = _TAPS - 1
        n = block.shape[0]
        self._tail_tmp.copy_(work[n:n + pad])
        work[:pad].copy_(self._tail_tmp)
        work[pad:pad + n].copy_(block)
        return work.view(1, 1, -1)

    @staticmethod
    def _conv(buf: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """对 _push 返回的视图做一次整块 FIR，输出长度 = 块长。"""
        return F.conv1d(buf, kernel.view(1, 1, -1)).view(-1)

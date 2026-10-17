"""原声直通流 — 三档运行态之一：把麦克风原声转发到主输出设备。

用途：实时变声经虚拟声卡输出，目标软件固定录 CABLE Output。原声档
开一条纯转发流（无模型、无 GPU、纯 memcpy），把麦克风原样写进主输出
（CABLE Input），用于临时说原话，目标软件全程不用换设备。

设计约束：
- 不依赖 torch/引擎：挂在 InferController 上，窗口启动阶段即可用。
- 延迟无关紧要：50ms 块纯 memcpy，CPU 占用可忽略。
- 采样率取输入设备默认值；WASAPI 共享模式下 Windows 音频引擎对输出侧
  做格式转换。打开失败时退回输出设备默认采样率再试一次，仍失败才报错。
- 运行中对 start() 幂等（已在跑则不动）；设备下拉框换选后要重开直通
  才生效（停止档 → 原声档 即重开）。
"""
import logging

import sounddevice as sd

from rvc.core.errors import AudioDeviceError

logger = logging.getLogger(__name__)


class PassthroughStream:
    """麦克风 → 主输出 的原声直通流（原声档运行态）。"""

    def __init__(self):
        self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self, in_dev: int, out_dev: int) -> None:
        """打开直通流（幂等：已在跑则直接返回）。失败抛 AudioDeviceError。"""
        if self.running:
            return
        in_rec = sd.query_devices(in_dev)
        out_rec = sd.query_devices(out_dev)
        in_max = int(in_rec["max_input_channels"])
        out_max = int(out_rec["max_output_channels"])
        if in_max <= 0:
            raise AudioDeviceError(
                f"直通输入设备「{in_rec.get('name', '?')}」不支持录音"
            )
        if out_max <= 0:
            raise AudioDeviceError(
                f"直通输出设备「{out_rec.get('name', '?')}」不支持播放"
            )
        in_ch = min(in_max, 2)
        out_ch = min(out_max, 2)

        # 采样率：先试输入设备默认值，失败退输出设备默认值（去重保序）
        last_err: Exception | None = None
        for sr in dict.fromkeys((int(in_rec["default_samplerate"]),
                                 int(out_rec["default_samplerate"]))):
            try:
                stream = sd.Stream(
                    device=(in_dev, out_dev),
                    samplerate=sr,
                    channels=(in_ch, out_ch),
                    dtype="float32",
                    blocksize=int(sr * 0.05),
                    callback=self._make_callback(in_ch, out_ch),
                )
                stream.start()
            except Exception as e:
                last_err = e
                continue
            self._stream = stream
            logger.info(
                "原声直通已开启：%s → %s（%dHz, %d→%dch）",
                in_rec.get("name", "?"), out_rec.get("name", "?"), sr, in_ch, out_ch,
            )
            return
        raise AudioDeviceError(f"原声直通流打开失败：{last_err}")

    @staticmethod
    def _make_callback(in_ch: int, out_ch: int):
        """生成通道映射回调：单声道广播到所有输出通道，多声道取前 out_ch 列。"""

        def _cb(indata, outdata, frames, time_info, status):
            # overflow/underflow 在直通链路无关紧要，静默忽略（避免日志刷屏）
            if in_ch == out_ch:
                outdata[:] = indata
            elif in_ch < out_ch:
                outdata[:] = indata  # numpy 广播：单声道 → 全部输出通道
            else:
                outdata[:] = indata[:, :out_ch]

        return _cb

    def stop(self) -> None:
        """安全中止并关闭直通流（幂等）。"""
        if self._stream is None:
            return
        try:
            self._stream.abort()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass
        self._stream = None
        logger.info("原声直通已关闭")

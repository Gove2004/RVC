"""音频流管理器 — 封装 sounddevice 设备查询、流创建/启动/停止、副输出。

从 RealtimeEngine 中拆分出的设备/流管理组件，负责：
- 设备查询与校验（输入/输出/副输出通道数）
- 主流创建与启动（sd.Stream）
- 副输出流创建与启动（sd.OutputStream + 队列）
- 流安全中止与关闭
- 设备日志打印

RealtimeEngine 保留对外接口，内部委托给本组件处理音频流。
"""
import logging
import queue

import numpy as np
import sounddevice as sd

from rvc.core.errors import AudioDeviceError

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """音频流管理器 — 管理 sounddevice 输入/输出/副输出流。"""

    def __init__(self):
        self.stream = None
        self.stream2 = None
        self.out2_q = queue.Queue(maxsize=10)
        self.enable_out2 = False
        self.sr = None
        self.channels = 1
        self.block_samples = 0

    def query_device(self, dev_idx: int) -> dict:
        """查询设备信息。"""
        return sd.query_devices(dev_idx)

    def validate_and_log_devices(self, in_dev: int, out_dev: int, out2_dev_idx: int | None = None) -> int:
        """校验输入/输出设备通道数，设置 channels，打印设备日志。

        Returns:
            channels: 最小通道数（min(输入, 输出, 2)）
        """
        in_info = self.query_device(in_dev)
        out_info = self.query_device(out_dev)
        in_max = int(in_info["max_input_channels"])
        out_max = int(out_info["max_output_channels"])

        if in_max <= 0:
            raise AudioDeviceError(
                f"输入设备不支持录音（max_input_channels={in_max}）："
                f"索引 {in_dev}「{in_info.get('name', '?')}」。请在设备设置中选择支持输入的设备。"
            )
        if out_max <= 0:
            raise AudioDeviceError(
                f"输出设备不支持播放（max_output_channels={out_max}）："
                f"索引 {out_dev}「{out_info.get('name', '?')}」。请在设备设置中选择支持输出的设备。"
            )

        self.channels = min(in_max, out_max, 2)

        logger.info("音频设备：")
        logger.info("  · 麦克风：%s", in_info.get('name', '?'))
        logger.info("  · 主输出：%s [%dch]", out_info.get('name', '?'), self.channels)

        if out2_dev_idx is not None:
            out2_info = self.query_device(out2_dev_idx)
            logger.info("  · 副输出：%s", out2_info.get('name', f"#{out2_dev_idx}"))

        return self.channels

    def validate_secondary_output(self, dev_idx: int, channels: int) -> None:
        """校验副输出设备通道数。"""
        out2_info = self.query_device(dev_idx)
        out2_max = int(out2_info["max_output_channels"])
        if out2_max < channels:
            raise AudioDeviceError(
                f"副输出设备「{out2_info.get('name', '?')}」只支持 {out2_max} 通道，"
                f"但主输出使用 {channels} 通道。请选择支持至少 {channels} 通道的副输出设备。"
            )

    def start_main_stream(self, callback, sr: int, channels: int, block_samples: int) -> None:
        """创建并启动主流。

        Args:
            callback: sounddevice 回调函数 (indata, outdata, frames, times, status)
            sr: 采样率
            channels: 通道数
            block_samples: 块大小（样本数）
        """
        self.sr = sr
        self.channels = channels
        self.block_samples = block_samples
        self.stream = sd.Stream(
            callback=callback,
            blocksize=block_samples,
            samplerate=sr,
            channels=channels,
            dtype="float32",
        )
        self.stream.start()

    def start_secondary_output(self, dev_idx: int, sr: int, channels: int, block_samples: int) -> None:
        """创建并启动副输出流。

        Args:
            dev_idx: 副输出设备索引
            sr: 采样率
            channels: 通道数
            block_samples: 块大小（样本数）
        """
        self.validate_secondary_output(dev_idx, channels)

        def out2_callback(outdata, frames, time_info, status):
            if not self.out2_q.empty():
                data = self.out2_q.get_nowait()
                outdata[:] = data[:frames]
            else:
                outdata[:] = 0

        self.stream2 = sd.OutputStream(
            device=dev_idx,
            samplerate=sr,
            channels=channels,
            dtype="float32",
            blocksize=block_samples,
            callback=out2_callback,
        )
        self.stream2.start()
        self.enable_out2 = True
        logger.debug("副输出流已启动: sr=%d, ch=%d, block=%d", sr, channels, block_samples)

        # 清空队列中可能残留的旧数据
        while not self.out2_q.empty():
            self.out2_q.get_nowait()

    def stop_all(self) -> None:
        """安全中止并关闭所有流（主流 + 副输出）。"""
        self._safe_close_stream(self.stream2)
        self._safe_close_stream(self.stream)
        self.stream = None
        self.stream2 = None
        self.enable_out2 = False

    @staticmethod
    def _safe_close_stream(stream) -> None:
        """安全中止并关闭 sounddevice 流，忽略所有异常。"""
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        """主流是否在运行。"""
        return self.stream is not None and self.stream.active

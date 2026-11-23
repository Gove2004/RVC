"""音频设备查询与校验的错误契约测试。

锁一条用户可见的契约：**设备不可用时的报错必须可读**。

背景：`sd.query_devices` 对越界/已拔出的索引抛 `PortAudioError`，原文是
「Error querying device N」——英文、无设备名、无处置建议。最典型的触发场景
是**设备热插拔后索引失效**（配置里存的是索引，拔掉 USB 声卡再启动就命中）。
`AudioStreams.query_device` 负责把它翻译成 `AudioDeviceError`，本测试防止
这层翻译被删掉后悄悄退回 PortAudio 原文。

不依赖真实声卡：用 mock 替换 `sd.query_devices`。
"""
import unittest
from unittest import mock

from rvc.core.errors import AudioDeviceError
from rvc.streaming.stream import AudioStreams


def _device(name: str, in_ch: int = 2, out_ch: int = 2) -> dict:
    return {"name": name, "max_input_channels": in_ch,
            "max_output_channels": out_ch, "index": 0}


class TestQueryDeviceErrorTranslation(unittest.TestCase):
    def test_out_of_range_index_becomes_readable_error(self):
        """越界索引：必须是 AudioDeviceError，且消息可读。"""
        streams = AudioStreams()
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        side_effect=Exception("Error querying device 9999")):
            with self.assertRaises(AudioDeviceError) as ctx:
                streams.query_device(9999)
        msg = str(ctx.exception)
        self.assertIn("9999", msg, "消息里要有出问题的索引")
        self.assertTrue(any("\u4e00" <= c <= "\u9fff" for c in msg),
                        "消息应是中文（与其它设备错误一致）")
        self.assertIn("重新选择", msg, "要有可操作的处置建议")

    def test_success_path_is_untouched(self):
        """正常查询必须原样返回设备信息，不引入包装层。"""
        streams = AudioStreams()
        rec = _device("扬声器")
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        return_value=rec):
            self.assertIs(streams.query_device(0), rec)


class TestValidateAndLogDevices(unittest.TestCase):
    def test_input_without_capture_channels_rejected(self):
        streams = AudioStreams()
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        side_effect=lambda i: _device("仅输出卡", 0, 2)):
            with self.assertRaises(AudioDeviceError) as ctx:
                streams.validate_and_log_devices(0, 1)
        self.assertIn("输入设备", str(ctx.exception))

    def test_output_without_playback_channels_rejected(self):
        streams = AudioStreams()
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        side_effect=lambda i: _device("仅输入卡", 2, 0)):
            with self.assertRaises(AudioDeviceError) as ctx:
                streams.validate_and_log_devices(0, 1)
        self.assertIn("输出设备", str(ctx.exception))

    def test_channels_is_min_of_both_capped_at_two(self):
        """channels = min(输入, 输出, 2)：单声道设备要退化到 1。"""
        streams = AudioStreams()
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        side_effect=lambda i: _device("单声道", 1, 1)):
            self.assertEqual(streams.validate_and_log_devices(0, 1), 1)

    def test_invalid_index_reported_as_device_error_not_portaudio(self):
        """校验入口的越界索引同样要变成可读错误（回归：曾漏出 PortAudioError）。"""
        streams = AudioStreams()
        with mock.patch("rvc.streaming.stream.sd.query_devices",
                        side_effect=Exception("Error querying device 9999")):
            with self.assertRaises(AudioDeviceError):
                streams.validate_and_log_devices(9999, 0)


if __name__ == "__main__":
    unittest.main()

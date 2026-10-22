"""rvc.core.config — 参数拷贝/序列化的全字段不变量。

防「protect 漏传事故」复发：任何给参数分组新增的字段，必须自动被
update_from / params_to_dict / params_from_dict 覆盖。本测试动态遍历
__dataclass_fields__ 对比全部字段，新增字段漏拷贝会立刻失败。
"""
import unittest

from gui.state.bindings import (
    GROUP_NAMES,
    TOP_LEVEL_FIELDS,
    params_from_dict,
    params_to_dict,
)
from rvc.core.config import InferenceParams, OfflineParams


def _snapshot(p: InferenceParams) -> dict:
    """拍下全部字段（含嵌套分组；不含派生属性 enable_out2）。"""
    out = {}
    for g in GROUP_NAMES:
        group = getattr(p, g)
        for f in group.__dataclass_fields__:
            if f == "enable_out2":
                continue
            out[f"{g}.{f}"] = getattr(group, f)
    for t in TOP_LEVEL_FIELDS:
        out[t] = getattr(p, t)
    return out


def _non_default() -> InferenceParams:
    """构造一份所有字段都偏离默认值的参数。"""
    p = InferenceParams()
    p.voice.formant = 0.37
    p.voice.pitch_map_src_min = 40.0
    p.voice.pitch_map_src_max = 640.0
    p.voice.pitch_map_dst_min = 120.0
    p.voice.pitch_map_dst_max = 720.0
    p.f0.method = "fcpe"
    p.f0.rmvpe_threshold = 0.04
    p.f0.fcpe_confidence_threshold = 0.02
    p.buffer.block_time = 0.1
    p.buffer.crossfade_time = 0.03
    p.buffer.extra_time = 1.5
    p.audio.sr_mode = "device"
    p.audio.hostapi = "MME"
    p.audio.input_device = "mic"
    p.audio.output_device = "spk"
    p.audio.output2_device = "不使用"
    p.audio.enable_out2 = True
    p.audio.device_index = 1
    p.rms_mix = 0.6
    p.volume = 1.7
    p.model_path = "X:/model.pth"
    p.hubert = "chinese"
    return p


class TestParamsCopy(unittest.TestCase):
    def test_update_from_copies_every_field(self):
        src = _non_default()
        dst = InferenceParams()
        dst.update_from(src)
        snap_src, snap_dst = _snapshot(src), _snapshot(dst)
        self.assertEqual(set(snap_src), set(snap_dst))
        for k, v in snap_src.items():
            self.assertEqual(snap_dst[k], v, f"update_from 漏拷字段: {k}")

    def test_from_inference_carries_every_field(self):
        src = _non_default()
        task = OfflineParams.from_inference(
            src, input_path="a.wav", output_path="b.wav"
        )
        snap_src, snap_task = _snapshot(src), _snapshot(task)
        self.assertEqual(set(snap_src), set(snap_task))
        for k, v in snap_src.items():
            self.assertEqual(snap_task[k], v, f"from_inference 漏拷字段: {k}")
        self.assertEqual(task.input_path, "a.wav")
        self.assertEqual(task.output_path, "b.wav")

    def test_dict_round_trip(self):
        p = _non_default()
        restored = params_from_dict(params_to_dict(p))
        snap_src, snap_rest = _snapshot(p), _snapshot(restored)
        self.assertEqual(set(snap_src), set(snap_rest))
        for k, v in snap_src.items():
            self.assertEqual(snap_rest[k], v, f"dict 往返丢失/篡改字段: {k}")

    def test_dict_round_trip_enable_out2_derived(self):
        # enable_out2 不持久化，从 output2_device 派生：真实设备名 → True
        p = _non_default()
        p.audio.output2_device = "HDMI 电视"
        restored = params_from_dict(params_to_dict(p))
        self.assertTrue(restored.audio.enable_out2)
        # 占位文本「不使用」→ False
        p.audio.output2_device = "不使用"
        self.assertFalse(params_from_dict(params_to_dict(p)).audio.enable_out2)

    def test_from_dict_missing_keys_use_defaults(self):
        # 缺键（旧配置文件）回落默认值，不抛异常
        p = params_from_dict({})
        d = InferenceParams()
        self.assertEqual(p.f0.method, d.f0.method)
        self.assertEqual(p.buffer.block_time, d.buffer.block_time)
        self.assertEqual(p.hubert, d.hubert)


if __name__ == "__main__":
    unittest.main()

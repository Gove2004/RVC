"""副输出队列（rvc.streaming.output.Out2Queue）行为测试。

重点锁两条实时性契约：
1. 队列满时丢最旧块，且被丢的缓冲回到空闲表（不会泄漏）；
2. 稳态 push/take 不再新分配缓冲——旧实现每块 outdata.copy() 一次 15-19KB。
"""
import unittest

import numpy as np

from rvc.streaming.output import Out2Queue


def _block(value: float, frames: int = 8, channels: int = 2) -> np.ndarray:
    return np.full((frames, channels), value, dtype=np.float32)


class TestOut2Queue(unittest.TestCase):
    def test_roundtrip_preserves_data(self):
        q = Out2Queue(maxsize=4)
        q.push(_block(0.25))
        out = np.zeros((8, 2), dtype=np.float32)
        self.assertTrue(q.take_into(out, 8))
        np.testing.assert_array_equal(out, _block(0.25))

    def test_take_on_empty_reports_false(self):
        q = Out2Queue(maxsize=4)
        out = np.ones((8, 2), dtype=np.float32)
        self.assertFalse(q.take_into(out, 8))
        # 交给调用方决定静音，本方法不写 outdata
        np.testing.assert_array_equal(out, np.ones((8, 2), dtype=np.float32))

    def test_full_queue_drops_oldest(self):
        q = Out2Queue(maxsize=3)
        for i in range(5):
            q.push(_block(float(i)))
        got = []
        out = np.zeros((8, 2), dtype=np.float32)
        while q.take_into(out, 8):
            got.append(float(out[0, 0]))
        self.assertEqual(got, [2.0, 3.0, 4.0])

    def test_buffers_recycled_steady_state(self):
        """长期 push/take 后空闲表大小不超过 maxsize+1（即总分配量恒定）。"""
        q = Out2Queue(maxsize=4)
        out = np.zeros((8, 2), dtype=np.float32)
        for i in range(500):
            q.push(_block(float(i % 7)))
            q.take_into(out, 8)
        # 一块在飞：push 取走空闲缓冲，take 拷完正好归还同一块
        self.assertEqual(len(q._free), 1)

    def test_dropped_block_buffer_returns_to_pool(self):
        q = Out2Queue(maxsize=2)
        out = np.zeros((8, 2), dtype=np.float32)
        for _ in range(10):
            q.push(_block(1.0))  # 队列满后每块都会触发一次丢弃
        self.assertLessEqual(len(q._free), q._q.maxsize + 1)
        # 队列里仍然只剩 maxsize 块
        drained = 0
        while q.take_into(out, 8):
            drained += 1
        self.assertEqual(drained, 2)

    def test_clear_releases_all(self):
        q = Out2Queue(maxsize=4)
        for i in range(4):
            q.push(_block(float(i)))
        q.clear()
        out = np.zeros((8, 2), dtype=np.float32)
        self.assertFalse(q.take_into(out, 8))
        # 队列空 + 空闲表持有被丢弃的 4 块
        self.assertEqual(len(q._free), 4)

    def test_block_size_change_reallocates(self):
        """块时长/通道数变化时旧缓冲作废，不产生形状错配。"""
        q = Out2Queue(maxsize=2)
        q.push(_block(1.0, frames=8, channels=2))
        q.push(_block(1.0, frames=16, channels=2))
        out = np.zeros((16, 2), dtype=np.float32)
        self.assertTrue(q.take_into(out, 16))
        self.assertEqual(out.shape, (16, 2))

    def test_short_block_padded_with_silence(self):
        """副输出块短于设备块长时补静音，而不是让 numpy 广播抛异常中断流。"""
        q = Out2Queue(maxsize=2)
        q.push(_block(0.5, frames=4, channels=2))
        out = np.ones((8, 2), dtype=np.float32)
        self.assertTrue(q.take_into(out, 8))
        np.testing.assert_array_equal(out[:4], _block(0.5, frames=4, channels=2))
        np.testing.assert_array_equal(out[4:], np.zeros((4, 2), dtype=np.float32))

    def test_push_does_not_hold_reference_to_caller_block(self):
        """入队是拷贝：调用方复用同一 outdata 缓冲时队列内容不被后续块覆盖。"""
        q = Out2Queue(maxsize=2)
        src = np.zeros((8, 2), dtype=np.float32)
        src[...] = 1.0
        q.push(src)
        src[...] = 2.0  # 模拟下一块写同一个设备缓冲
        out = np.zeros((8, 2), dtype=np.float32)
        q.take_into(out, 8)
        np.testing.assert_array_equal(out, np.ones((8, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

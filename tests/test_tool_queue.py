"""模型工具队列契约 — 串行执行但不阻塞 GUI 线程。

契约：正在跑就排队、立刻返回，轮到它再启动——在主线程 `wait(10000)` 等
上一个工具线程会把 GUI 僵住 10 秒。
"""
import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from gui.controller.tool_controller import run_tool_async

app = QApplication.instance() or QApplication([])


def _pump_until(predicate, timeout=10.0) -> bool:
    """跑事件循环直到 predicate 成立（QThread 信号要靠事件循环投递）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestToolQueue(unittest.TestCase):
    def setUp(self):
        self.win = QWidget()
        self.win.tool_runner = None
        self.win.tool_queue = []
        self.calls = []

    def tearDown(self):
        if self.win.tool_runner is not None and self.win.tool_runner.isRunning():
            self.win.tool_runner.wait(5000)
        app.processEvents()
        self.win.deleteLater()

    def test_second_request_returns_without_blocking(self):
        gate = threading.Event()
        run_tool_async(self.win, gate.wait, 5,
                       on_done=lambda ok, msg: self.calls.append("first"))

        started = time.monotonic()
        run_tool_async(self.win, lambda: "second",
                       on_done=lambda ok, msg: self.calls.append("second"))
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5,
                        "已有任务在跑时，第二次调用必须立刻返回而不是主线程等待")
        self.assertEqual(len(self.win.tool_queue), 1, "第二个请求应排队")
        self.assertEqual(self.calls, [], "任务都还没结束，回调不该触发")

        gate.set()
        self.assertTrue(_pump_until(lambda: len(self.calls) == 2))
        self.assertEqual(self.calls, ["first", "second"], "排队任务必须按提交顺序执行")

    def test_queue_is_empty_after_drain(self):
        results = []
        for i in range(3):
            run_tool_async(self.win, lambda i=i: f"r{i}",
                           on_done=lambda ok, msg, i=i: results.append(i))

        self.assertTrue(_pump_until(lambda: len(results) == 3))
        self.assertEqual(results, [0, 1, 2])
        self.assertEqual(self.win.tool_queue, [])
        self.assertIsNone(self.win.tool_runner, "全部结束后不该残留 runner 引用")

    def test_failure_is_reported_to_caller(self):
        seen = []

        def boom():
            raise RuntimeError("合并失败")

        run_tool_async(self.win, boom, on_done=lambda ok, msg: seen.append((ok, msg)))
        self.assertTrue(_pump_until(lambda: seen))
        self.assertEqual(seen[0][0], False)
        self.assertIn("合并失败", seen[0][1])


if __name__ == "__main__":
    unittest.main()

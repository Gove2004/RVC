"""controller 层无头测试 — OfflineConversion / DeviceCatalog 契约行为。

controller 已契约化（只依赖 OfflineViewHost / DeviceViewHost），用 FakeHost
替身即可测试业务逻辑，不需要任何 Qt 控件。
"""
import unittest
from unittest import mock


class FakeHost:
    """OfflineViewHost / DeviceViewHost 替身 — 记录全部语义调用。"""

    def __init__(self, input_path="", model_path="m.pth", hubert="chinese",
                 running=False, exists=None, loading=False):
        self.controller = mock.Mock()
        self.controller.is_running = running
        self.controller.passthrough_running = False
        self.controller.engine = object()
        self._loading = loading
        self._input_path = input_path
        self._model_path = model_path
        self._hubert = hubert
        self._exists = exists if exists is not None else (lambda p: bool(p))
        self.warnings = []
        self.errors = []
        self.calls = []
        self.confirm_answer = True
        self.params = object()  # collect_gui_state 返回值（ OfflineParams.from_inference 前会被打桩）

    # 读取
    def offline_input_path(self):
        return self._input_path

    def offline_model_path(self):
        return self._model_path

    def offline_hubert_variant(self):
        return self._hubert

    def is_engine_loading(self):
        return self._loading

    def collect_gui_state(self):
        return self.params

    # 反馈
    def show_warning(self, m):
        self.warnings.append(m)

    def show_error(self, m):
        self.errors.append(m)

    def offline_confirm_delete(self, path):
        self.calls.append(("confirm_delete", path))
        return self.confirm_answer

    # 转换态 UI
    def offline_enter_converting(self):
        self.calls.append(("enter_converting",))

    def offline_enter_cancelling(self):
        self.calls.append(("enter_cancelling",))

    def offline_reset(self, status):
        self.calls.append(("reset", status))

    def offline_set_progress(self, current, total):
        self.calls.append(("progress", current, total))

    def offline_set_status(self, text):
        self.calls.append(("status", text))

    def offline_set_previewing(self, on, name=""):
        self.calls.append(("previewing", on, name))

    def last_call(self, name):
        found = [c for c in self.calls if c[0] == name]
        return found[-1] if found else None


class _FakeSignal:
    def connect(self, *a):
        pass


class _FakeWorker:
    """OfflineWorker 替身 — 记录构造参数，不启动线程。"""

    instances = []

    def __init__(self, params, engine_provider, parent=None):
        self.params = params
        self.engine_provider = engine_provider
        self.parent = parent
        self.progress = _FakeSignal()
        self.result_ready = _FakeSignal()
        self.error = _FakeSignal()
        self.cancelled = _FakeSignal()
        # QThread 内建 finished（真实类由 Qt 提供；控制器用它挂 deleteLater 回收）
        self.finished = _FakeSignal()
        self.started = False
        _FakeWorker.instances.append(self)

    def start(self):
        self.started = True

    def deleteLater(self):
        """QObject.deleteLater 替身（真类由 Qt 提供）。"""


def _make_conversion(host):
    from gui.controller.offline_controller import OfflineConversion

    return OfflineConversion(host)


class TestStartConversionGuards(unittest.TestCase):
    """start_conversion 的四道守卫：互斥 → 输入 → 存在性 → 模型。"""

    def _patch_worker(self):
        _FakeWorker.instances = []
        return mock.patch("gui.controller.offline_controller.OfflineWorker", _FakeWorker)

    def test_realtime_running_blocks(self):
        host = FakeHost(input_path="a.wav", running=True)
        conv = _make_conversion(host)
        with self._patch_worker():
            conv.start_conversion()
        self.assertIn("实时推理正在运行", host.warnings[-1])
        self.assertIsNone(host.last_call("enter_converting"))

    def test_missing_input_blocks(self):
        host = FakeHost(input_path="")
        conv = _make_conversion(host)
        with self._patch_worker():
            conv.start_conversion()
        self.assertIn("请选择输入文件", host.warnings[-1])

    def test_nonexistent_input_blocks(self):
        host = FakeHost(input_path="E:/no/such.wav")
        host._exists = lambda p: False
        conv = _make_conversion(host)
        with mock.patch("gui.controller.offline_controller.os.path.exists", host._exists):
            with self._patch_worker():
                conv.start_conversion()
        self.assertIn("文件不存在", host.warnings[-1])

    def test_missing_model_blocks(self):
        host = FakeHost(input_path="a.wav", model_path="")
        conv = _make_conversion(host)
        with mock.patch("gui.controller.offline_controller.os.path.exists", lambda p: True):
            with self._patch_worker():
                conv.start_conversion()
        self.assertIn("请先在「音色」中选择模型文件", host.warnings[-1])


class TestStartConversionHappyPath(unittest.TestCase):
    def test_start_creates_worker_and_enters_converting(self):
        host = FakeHost(input_path="E:/audio/aaa.wav", model_path="m.pth")
        conv = _make_conversion(host)
        with mock.patch("gui.controller.offline_controller.os.path.exists", lambda p: True), \
             mock.patch("gui.controller.offline_controller.OfflineParams.from_inference") as from_inf, \
             mock.patch("gui.controller.offline_controller.OfflineWorker", _FakeWorker):
            from_inf.return_value = mock.Mock()
            conv.start_conversion()
        self.assertEqual(len(_FakeWorker.instances), 1)
        worker = _FakeWorker.instances[0]
        self.assertTrue(worker.started)
        # 输出路径 = 输入同目录 + _converted.wav（from_inference 的实参）
        kwargs = from_inf.call_args.kwargs
        self.assertEqual(kwargs["output_path"], "E:/audio/aaa_converted.wav")
        self.assertEqual(kwargs["input_path"], "E:/audio/aaa.wav")
        # hubert 不单独传：它随 collect_gui_state 一起进入 from_inference
        self.assertNotIn("hubert", kwargs)
        self.assertTrue(conv.is_converting)
        self.assertIsNotNone(host.last_call("enter_converting"))

    def test_cancel_when_idle_is_noop(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv.cancel_conversion()
        self.assertEqual(host.calls, [])


class TestProgressAndCompletion(unittest.TestCase):
    def test_on_progress_forwards(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv._on_progress(3, 10)
        self.assertEqual(host.last_call("progress"), ("progress", 3, 10))

    def test_finished_resets(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv._converting = True
        conv._on_finished("out.wav")
        self.assertFalse(conv.is_converting)
        # 状态行必须带上 result_ready 携带的路径，否则只说「完成」，
        # 用户无从确认这次产出的是哪个文件
        self.assertEqual(host.last_call("reset"), ("reset", "完成: out.wav"))

    def test_error_resets_and_shows(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv._on_error("boom")
        self.assertEqual(host.last_call("reset"), ("reset", "错误"))
        self.assertIn("离线推理错误", host.errors[-1])

    def test_cancel_enter_cancelling(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv.worker = mock.Mock()
        conv._converting = True
        conv.cancel_conversion()
        conv.worker.request_cancel.assert_called_once()
        self.assertIsNotNone(host.last_call("enter_cancelling"))


class TestWorkerLifecycle(unittest.TestCase):
    """worker 生命周期 — 转换结束后必须立刻解引用。

    线程对象由 finished→deleteLater 回收，但控制器若还留着引用，
    下一次 shutdown()/cancel_conversion() 就会撞上已销毁的 C++ 对象
    （PySide6 抛 RuntimeError）。退出路径上一旦抛出，后面的保存配置/停引擎/
    quit 全部不执行 → 托盘图标已隐藏、进程仍在跑。
    """

    def test_result_handlers_release_worker(self):
        for handler, arg in (("_on_finished", "out.wav"), ("_on_error", "boom"), ("_on_cancelled", None)):
            with self.subTest(handler=handler):
                conv = _make_conversion(FakeHost())
                conv.worker = mock.Mock()
                conv._converting = True
                if arg is None:
                    getattr(conv, handler)()
                else:
                    getattr(conv, handler)(arg)
                self.assertIsNone(conv.worker, "收尾后必须解引用，否则 shutdown 会访问已销毁对象")
                self.assertFalse(conv.is_converting)

    def test_shutdown_tolerates_deleted_worker(self):
        """C++ 对象已被 deleteLater 销毁时 shutdown 不得抛异常。"""
        conv = _make_conversion(FakeHost())
        dead = mock.Mock()
        dead.isRunning.side_effect = RuntimeError("Internal C++ object already deleted")
        conv.worker = dead
        conv.shutdown()  # 不抛即通过
        self.assertIsNone(conv.worker)

    def test_shutdown_cancels_running_worker(self):
        conv = _make_conversion(FakeHost())
        worker = mock.Mock()
        worker.isRunning.return_value = True
        worker.wait.return_value = True
        conv.worker = worker
        conv._converting = True
        conv.shutdown()
        worker.request_cancel.assert_called_once()
        self.assertFalse(conv.is_converting)

    def test_shutdown_without_worker_is_noop(self):
        conv = _make_conversion(FakeHost())
        conv.shutdown()  # 不抛即通过
        self.assertIsNone(conv.worker)


class TestPreviewGuard(unittest.TestCase):
    """监听必须避开实时运行：sd.play 不指定设备，会落到 setup 改写的默认设备上。"""

    def test_preview_blocked_while_realtime_running(self):
        host = FakeHost(input_path="a.wav", running=True)
        conv = _make_conversion(host)
        conv._preview_active = lambda: False  # 不碰 PortAudio
        conv.toggle_preview()
        self.assertIn("监听会把文件直接播到输出设备", host.warnings[-1])
        self.assertIsNone(host.last_call("previewing"))

    def test_preview_blocked_while_passthrough(self):
        host = FakeHost(input_path="a.wav")
        host.controller.passthrough_running = True
        conv = _make_conversion(host)
        conv._preview_active = lambda: False
        conv.toggle_preview()
        self.assertIn("监听会把文件直接播到输出设备", host.warnings[-1])


class TestCancelFeedback(unittest.TestCase):
    def test_progress_does_not_overwrite_cancelling_status(self):
        host = FakeHost()
        conv = _make_conversion(host)
        conv.worker = mock.Mock()
        conv._converting = True
        conv.cancel_conversion()
        conv._on_progress(30, 100)
        # 进度条照常更新，但状态行必须停在「正在取消...」
        self.assertEqual(host.last_call("progress"), ("progress", 30, 100))
        self.assertIsNone(host.last_call("status"))


class TestDeleteOutput(unittest.TestCase):
    def _conv_with_output(self, host, path="E:/audio/aaa.wav"):
        from gui.controller.offline_controller import OUTPUT_SUFFIX

        out = path.replace(".wav", OUTPUT_SUFFIX)
        host._input_path = path
        return _make_conversion(host), out

    @mock.patch("gui.controller.offline_controller.recycle_file")
    @mock.patch("gui.controller.offline_controller.os.path.exists", lambda p: True)
    def test_delete_confirmed_recycles(self, recycle):
        host = FakeHost()
        conv, out = self._conv_with_output(host)
        conv.delete_output()
        recycle.assert_called_once_with(out)
        self.assertEqual(host.last_call("status"), ("status", "输出文件已移入回收站"))

    @mock.patch("gui.controller.offline_controller.recycle_file")
    @mock.patch("gui.controller.offline_controller.os.path.exists", lambda p: True)
    def test_delete_declined_keeps_file(self, recycle):
        host = FakeHost()
        host.confirm_answer = False
        conv, _ = self._conv_with_output(host)
        conv.delete_output()
        recycle.assert_not_called()
        # 只发生确认交互，无回收站调用、无状态行更新
        self.assertEqual([c[0] for c in host.calls], ["confirm_delete"])

    @mock.patch("gui.controller.offline_controller.os.path.exists", lambda p: False)
    def test_delete_missing_output_warns(self):
        host = FakeHost()
        conv, _ = self._conv_with_output(host)
        conv.delete_output()
        self.assertIn("输出文件不存在", host.warnings[-1])


class TestDefaultOutputPath(unittest.TestCase):
    def test_suffix(self):
        from gui.controller.offline_controller import default_output_path

        self.assertEqual(default_output_path("E:/x/y.wav"), "E:/x/y_converted.wav")
        self.assertEqual(default_output_path(""), "")


class TestDeviceCatalog(unittest.TestCase):
    """DeviceCatalog 的数据/决策逻辑（get_audio_devices 打桩，不碰 PortAudio）。"""

    def _catalog(self, selections=("WASAPI", "mic", "spk", "不使用")):
        from gui.controller.device_controller import DeviceCatalog

        host = mock.Mock()
        host.device_current_selections.return_value = selections
        return DeviceCatalog(host), host

    @mock.patch("gui.controller.device_controller.get_audio_devices")
    def test_refresh_passes_prev_hostapi_and_updates_maps(self, gad):
        gad.return_value = (["WASAPI", "MME"], ["mic"], ["spk", "tv"], [3, 7], [4, 9])
        cat, host = self._catalog()
        cat._refresh_devices()
        # 以当前 hostapi 文本为过滤条件
        gad.assert_called_once_with("WASAPI")
        # 索引映射更新
        self.assertEqual(cat.get_input_device_index(1), 7)
        self.assertEqual(cat.get_output_device_index(1), 9)
        self.assertEqual(cat.get_output_device_index(-1), -1)
        # 宿主收到重填指令（含原选择）
        host.device_populate.assert_called_once_with(
            ["WASAPI", "MME"], ["mic"], ["spk", "tv"], ("WASAPI", "mic", "spk", "不使用"))

    @mock.patch("gui.controller.device_controller.get_audio_devices")
    def test_index_out_of_range_raises(self, gad):
        gad.return_value = (["WASAPI"], ["mic"], ["spk"], [3], [4])
        cat, _ = self._catalog()
        cat._refresh_devices()
        with self.assertRaises(IndexError):
            cat.get_input_device_index(5)
        with self.assertRaises(IndexError):
            cat.get_output_device_index(5)


if __name__ == "__main__":
    unittest.main()

"""GUI 页签构建冒烟 — 离屏构造不崩 + 关键不变量。

只测构造期可判定的稳定契约，不断言布局细节（页签迭代频繁，易碎）。
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from types import SimpleNamespace

from rvc.core.config import InferenceParams

app = QApplication.instance() or QApplication([])


class _Win:
    runtime_params = InferenceParams()


class TestTabsConstruct(unittest.TestCase):
    def test_params_tab_builds(self):
        from gui.view.tabs.params_tab import build_params_tab

        w = build_params_tab(_Win())
        self.assertIsNotNone(w.layout())

    def test_details_tab_no_inspect_button(self):
        # 详情页契约：选定即自动解析，没有「查看」按钮
        from gui.view.tabs.details_tab import build_details_tab

        class _Ctrl:
            models_dir = "."
        class _Tools:
            def inspect_model(self, p):
                return ""
        class _Win3(QWidget):
            def __init__(self):
                super().__init__()
                self.controller = _Ctrl()
                self.model_tools = _Tools()

        w = build_details_tab(_Win3())
        btns = [b.text() for b in w.findChildren(QPushButton)]
        self.assertEqual(btns, ["选择模型..."])


class TestTimbreTabFormant(unittest.TestCase):
    """性别因子滑条：量程来自引擎常量，且运行中不禁用。"""

    def _build(self):
        from types import SimpleNamespace

        from rvc.core.config import (
            FORMANT_MAX, FORMANT_MIN, VOLUME_MAX, VOLUME_MIN, InferenceParams,
        )
        from gui.view.tabs.timbre_tab import build_timbre_tab

        class _Win(QWidget):
            runtime_params = InferenceParams()
            controller = SimpleNamespace(
                models_dir=".",
                formant_range=(FORMANT_MIN, FORMANT_MAX),
                volume_range=(VOLUME_MIN, VOLUME_MAX),
                hubert_variants=("base", "chinese"),
            )

        win = _Win()
        win._page = build_timbre_tab(win)   # 页容器必须保活，否则其子控件随之销毁
        return win

    def test_slider_spans_engine_range(self):
        from rvc.core.config import FORMANT_MAX, FORMANT_MIN

        win = self._build()   # 持有窗口与页容器引用：子控件随它们销毁
        slider = win.formant_slider
        slider.setValue(FORMANT_MIN)
        self.assertAlmostEqual(slider.value(), FORMANT_MIN, places=6)
        slider.setValue(FORMANT_MAX)
        self.assertAlmostEqual(slider.value(), FORMANT_MAX, places=6)
        slider.setValue(FORMANT_MAX + 99)   # 越界钳到量程上界
        self.assertAlmostEqual(slider.value(), FORMANT_MAX, places=6)

    def test_running_mode_does_not_disable_slider(self):
        """变声中必须能拖性别因子——各档位的合成器图已在启动时预捕获。"""
        from types import SimpleNamespace

        from gui.view.window import MainWindow

        win = MainWindow.__new__(MainWindow)
        win._loading = False
        win.controller = SimpleNamespace(is_running=True, passthrough_running=False)
        win.mode_toggle = SimpleNamespace(set_mode=lambda _m: None)

        def _boom(_v):
            raise AssertionError("变声运行中不许禁用性别因子滑条")

        win.formant_slider = SimpleNamespace(setEnabled=_boom)
        win._refresh_mode()


class TestAudioFilterCoversEngineExts(unittest.TestCase):
    """文件对话框的扩展名过滤器必须覆盖引擎认识的所有音频格式。"""

    def test_no_drift(self):
        from rvc.core.constants import AUDIO_EXTS

        from gui.view.tabs.offline_tab import AUDIO_FILTER
        for ext in AUDIO_EXTS:
            self.assertIn(f"*{ext}", AUDIO_FILTER,
                          f"引擎支持 {ext}，但文件对话框选不了它")


class TestNoRealismSliderLeftover(unittest.TestCase):
    """真实感滑条已随整条链删除——两页都不该再有它，防止残留控件回来。"""

    def test_params_tab_has_no_realism_slider(self):
        from gui.view.tabs.params_tab import build_params_tab

        class _Win:
            runtime_params = InferenceParams()

        win = _Win()
        win._page = build_params_tab(win)
        self.assertFalse(hasattr(win, "realism_slider"))

    def test_timbre_tab_has_no_realism_slider(self):
        from types import SimpleNamespace

        from rvc.core.config import (
            FORMANT_MAX, FORMANT_MIN, VOLUME_MAX, VOLUME_MIN, InferenceParams,
        )
        from gui.view.tabs.timbre_tab import build_timbre_tab

        class _Win(QWidget):
            runtime_params = InferenceParams()
            controller = SimpleNamespace(
                models_dir=".",
                formant_range=(FORMANT_MIN, FORMANT_MAX),
                volume_range=(VOLUME_MIN, VOLUME_MAX),
                hubert_variants=("base", "chinese"),
            )

        win = _Win()
        win._page = build_timbre_tab(win)
        self.assertFalse(hasattr(win, "realism_slider"))


class TestDropdowns(unittest.TestCase):
    """toggle→下拉框：语义 userData、切换槽副作用。

    下拉框的 userData（rmvpe/fcpe、model/device）是 collect/apply 的命中键，
    改文案或漏挂 userData 会静默落空首项——这里把语义值钉死。
    """

    def _params_win(self):
        from gui.view.tabs.params_tab import build_params_tab

        win = _Win()
        win._page = build_params_tab(win)   # 页容器保活，子控件才不随之销毁
        return win

    def _device_win(self):
        from gui.view.tabs.device_tab import build_device_tab

        class _DWin:
            def _ha_changed(self, *_a):
                pass

        win = _DWin()
        win._page, win._refresh = build_device_tab(win)
        return win

    def test_f0_combo_semantic_userdata(self):
        win = self._params_win()
        combo = win.f0_combo
        items = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        self.assertEqual(items, [("RMVPE", "rmvpe"), ("FCPE", "fcpe")])

    def test_sr_combo_semantic_userdata(self):
        win = self._device_win()
        combo = win.sr_combo
        items = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]
        self.assertEqual(items, [("模型", "model"), ("设备", "device")])

    def test_f0_switch_slot_refreshes_threshold(self):
        from gui.view.window import MainWindow

        pw = self._params_win()
        win = MainWindow.__new__(MainWindow)
        win.runtime_params = InferenceParams()
        win.f0_combo = pw.f0_combo
        win.f0_threshold_slider = pw.f0_threshold_slider
        win.f0_threshold_name = pw.f0_threshold_name

        win.f0_combo.setCurrentIndex(1)   # fcpe
        win._on_f0_method_changed()
        self.assertEqual(win.runtime_params.f0.method, "fcpe")
        self.assertEqual(win.f0_threshold_name.text(), "FCPE 阈值")
        self.assertAlmostEqual(win.f0_threshold_slider.value(),
                               win.runtime_params.f0.fcpe_confidence_threshold, places=6)

        win.f0_combo.setCurrentIndex(0)   # rmvpe
        win._on_f0_method_changed()
        self.assertEqual(win.runtime_params.f0.method, "rmvpe")
        self.assertEqual(win.f0_threshold_name.text(), "RMVPE 阈值")
        self.assertAlmostEqual(win.f0_threshold_slider.value(),
                               win.runtime_params.f0.rmvpe_threshold, places=6)

    def test_sr_switch_slot_writes_userdata(self):
        from gui.view.window import MainWindow

        dw = self._device_win()
        win = MainWindow.__new__(MainWindow)
        win.runtime_params = InferenceParams()
        win.sr_combo = dw.sr_combo

        win.sr_combo.setCurrentIndex(1)
        win._on_sr_mode_changed()
        self.assertEqual(win.runtime_params.audio.sr_mode, "device")
        win.sr_combo.setCurrentIndex(0)
        win._on_sr_mode_changed()
        self.assertEqual(win.runtime_params.audio.sr_mode, "model")


class TestRangeSliderClampInvariant(unittest.TestCase):
    """RangeSlider.set_selected_range 必须始终满足 min <= low <= high <= max。

    旧实现是「先各自钳制、后交换」，交换后的 low 是那个只被上界钳过、从未被
    下界钳过的值，于是 low 能逃出量程下界（实测 (1200,500) -> low=500/high=1200，
    (0,0) -> low=0 而控件下界是 10）。这条路径只有越界或倒置的配置会走到
    （拖拽的 _update_low/_update_high 钳制是对的），但 set_selected_range 会
    发射 rangeChanged，而 window 把它接到写 runtime_params 的回调上——于是
    **加载配置这个动作本身**就把坏值写进运行期参数并随下次保存落盘，静默且
    自我固化。此前 tests/ 对 RangeSlider 零覆盖。
    """

    LO, HI = 10.0, 1000.0

    def _slider(self):
        from gui.view.widgets.sliders import RangeSlider

        return RangeSlider(self.LO, self.HI, 10.0, self.LO, self.HI)

    def test_invariant_holds_for_out_of_range_and_inverted_pairs(self):
        s = self._slider()
        vals = [-50.0, 0.0, 5.0, 9.0, 10.0, 30.0, 100.0, 500.0,
                800.0, 1000.0, 1200.0]
        for a in vals:
            for b in vals:
                with self.subTest(pair=(a, b)):
                    s.set_selected_range(a, b)
                    low, high = s.low(), s.high()
                    self.assertGreaterEqual(low, self.LO, f"({a},{b}) low 逃出下界")
                    self.assertLessEqual(high, self.HI, f"({a},{b}) high 逃出上界")
                    self.assertLessEqual(low, high, f"({a},{b}) low > high")

    def test_inverted_pair_is_swapped_not_clamped_to_degenerate(self):
        s = self._slider()
        s.set_selected_range(500.0, 10.0)
        self.assertEqual((s.low(), s.high()), (10.0, 500.0))

    def test_valid_pair_is_untouched(self):
        """正常值不得被钳制改动（修复不能引入新的偏移）。"""
        s = self._slider()
        s.set_selected_range(100.0, 800.0)
        self.assertEqual((s.low(), s.high()), (100.0, 800.0))


class TestPitchMapPairSanitised(unittest.TestCase):
    """加载配置时倒置的音域区间要被收敛，否则 apply_pitch_map 静默失效。

    apply_pitch_map 对 src_min >= src_max 或 dst_min >= dst_max 的区间
    **原样返回**（映射整个失效），而存档是明文 JSON、可被手改。
    """

    def test_sorted_pair_swaps_inverted(self):
        from gui.state.bindings import _sorted_pair

        self.assertEqual(_sorted_pair(500, 10), (10.0, 500.0))
        self.assertEqual(_sorted_pair(10, 500), (10.0, 500.0))

    def test_sorted_pair_preserves_equal(self):
        from gui.state.bindings import _sorted_pair

        self.assertEqual(_sorted_pair(0, 0), (0.0, 0.0))

    def test_sorted_pair_returns_floats(self):
        from gui.state.bindings import _sorted_pair

        lo, hi = _sorted_pair(10, 500)
        self.assertIsInstance(lo, float)
        self.assertIsInstance(hi, float)


class TestStopDuringLoadingAlsoStopsEngine(unittest.TestCase):
    """_stop() 在 _loading 期间也必须停引擎。

    _loading 为 True 的区间覆盖 _on_loaded 执行期：_on_loaded 调 _mark_running()
    之后引擎**已经在跑**，而 _loading 要等 _on_load_done 才置 False。旧写法在
    _loading 分支里只取消加载就 return，会出现「UI 显示已停止、麦克风仍在往
    虚拟声卡送声」。stop() 是幂等 no-op，无条件调用是安全的。
    """

    def _lifecycle(self):
        from gui.view.lifecycle import WindowLifecycle

        lc = WindowLifecycle.__new__(WindowLifecycle)

        class _Ctrl:
            def __init__(self):
                self.stop_calls = 0
                self.is_running = True

            def stop(self):
                self.stop_calls += 1

        lc.controller = _Ctrl()
        lc._loading = True
        lc.tray = None
        lc._cancel_load = lambda: None
        lc._reset_runtime_ui = lambda: None
        return lc

    def test_loading_branch_calls_controller_stop(self):
        lc = self._lifecycle()
        lc._stop()
        self.assertEqual(lc.controller.stop_calls, 1,
                         "_loading 期间点停止没停引擎——麦克风会继续送声")

    def test_not_running_short_circuits(self):
        """引擎没跑时不该白调 stop（保持原有短路语义）。"""
        lc = self._lifecycle()
        lc._loading = False
        lc.controller.is_running = False
        lc._stop()
        self.assertEqual(lc.controller.stop_calls, 0)

    def test_running_branch_calls_stop(self):
        lc = self._lifecycle()
        lc._loading = False
        lc.controller.is_running = True
        lc._stop()
        self.assertEqual(lc.controller.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()

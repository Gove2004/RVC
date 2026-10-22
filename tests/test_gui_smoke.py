"""GUI 页签构建冒烟 — 离屏构造不崩 + 关键不变量。

只测构造期可判定的稳定契约，不断言布局细节（页签迭代频繁，易碎）。
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton, QWidget

app = QApplication.instance() or QApplication([])


class _F0:
    method = "fcpe"
    rmvpe_threshold = 0.05
    fcpe_confidence_threshold = 0.03


class _P:
    f0 = _F0()


class _Win:
    runtime_params = _P()


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


if __name__ == "__main__":
    unittest.main()

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
        # 详情页契约：选定即自动解析，不再有「查看」按钮
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


if __name__ == "__main__":
    unittest.main()

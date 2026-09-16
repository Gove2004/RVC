"""GUI 冒烟测试（S10 自证）— offscreen 构造窗口 + save_state 往返等价。

替代"逐页签人工清单走查"的自动化基线：
- infer/train 双窗口可完整装配（offscreen），全部页签控件构建成功
- collect_gui_state → params_to_dict → params_from_dict → apply_gui_state →
  collect_gui_state 往返一致（save_state 序列化语义等价）
- TrainGuiState dict 往返一致

托盘在 offscreen 下不可用，用桩替换（真实托盘行为不在本测试范围）。
运行环境：QT_QPA_PLATFORM=offscreen 由测试自行设置（必须在 QApplication
构造前生效，故本文件不与其他 Qt 测试共享进程内状态）。
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

# offscreen 无系统托盘：给 TrayManager 打桩，避免窗口构造走 sys.exit(1)
import gui.infer.view.tray as _tray_mod  # noqa: E402


class _StubTray:
    def __init__(self, window, on_quit=None, **kwargs):
        self.window = window
        self.on_quit = on_quit

    def update_status(self):
        pass

    def notify_minimized(self):
        pass


_tray_mod.TrayManager = _StubTray

from gui.infer.state.bindings import params_from_dict, params_to_dict  # noqa: E402
from gui.infer.view.window import MainWindow as InferWindow  # noqa: E402
from gui.train.state.train_state import TrainGuiState  # noqa: E402
from gui.train.view.window import TrainWindow  # noqa: E402


class TestInferWindowRoundtrip(unittest.TestCase):
    def test_window_assembles_and_state_roundtrip(self):
        win = InferWindow()
        try:
            # 全部页签控件存在（装配完整性抽查）
            for attr in ("btn_toggle", "pitch_lbl", "delay_lbl",
                         "hostapi_combo", "input_combo", "output_combo", "output2_combo",
                         "sr_model_radio", "sr_device_radio", "f0_rmvp_btn", "f0_fcpe_btn",
                         "model_path_btn", "hubert_combo",
                         "block_time_slider", "crossfade_slider", "extra_time_slider",
                         "rms_mix_slider", "formant_slider",
                         "exp_f0_threshold_slider", "exp_pitch_map_src_range",
                         "exp_pitch_map_dst_range",
                         "offline_input_btn", "offline_output", "offline_button"):
                self.assertTrue(hasattr(win, attr), f"控件缺失: {attr}")

            state1 = win.collect_gui_state()
            # 序列化 → 反序列化 → 回填控件 → 再收集，必须等价
            restored = params_from_dict(params_to_dict(state1))
            win.apply_gui_state(restored)
            state2 = win.collect_gui_state()
            self.assertEqual(params_to_dict(state1), params_to_dict(state2))
        finally:
            win._timer.stop()
            win.close()
            win.deleteLater()


class TestTrainWindowRoundtrip(unittest.TestCase):
    def test_window_assembles_and_state_roundtrip(self):
        win = TrainWindow()
        try:
            for attr in ("exp_name", "input_dir", "sample_rate", "hubert",
                         "epochs", "batch_size", "save_every", "learning_rate",
                         "pretrain_g", "pretrain_d",
                         "btn_preprocess", "btn_f0", "btn_feature", "btn_train",
                         "btn_all", "stop_btn",
                         "stage_label", "epoch_label", "loss_label", "progress_bar", "log_edit",
                         "merge_a", "merge_b", "merge_slider",
                         "inspect_path", "fix_path", "fix_info"):
                self.assertTrue(hasattr(win, attr), f"控件缺失: {attr}")

            state1 = win.collect_gui_state()
            win.apply_gui_state(TrainGuiState.from_dict(state1.to_dict()))
            state2 = win.collect_gui_state()
            self.assertEqual(state1.to_dict(), state2.to_dict())

            # 采样率切换 → 底模路径映射（TrainController 业务，走 controller）
            win._on_sr_changed("48k")
        finally:
            win.close()
            win.deleteLater()


class TestTrainGuiStateDefaults(unittest.TestCase):
    def test_dict_roundtrip_with_defaults(self):
        state = TrainGuiState.from_dict({})
        self.assertEqual(state.to_dict(), TrainGuiState().to_dict())


if __name__ == "__main__":
    unittest.main()

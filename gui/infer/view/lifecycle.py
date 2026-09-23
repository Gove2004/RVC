"""推理 GUI 生命周期 — MainWindow 的生命周期职责单元（D5 拆分）。

职责：配置持久化、引擎预热、启动/停止（加载线程管理）、
运行时错误处理、托盘退出、关闭行为。UI 装配与信号绑定在 window.py。

Mixin 形式：方法挂在 MainWindow 上，通过 self 访问窗口的控件与 controller。
"""
import logging

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from gui.infer.view.widgets import LoadThread

logger = logging.getLogger(__name__)


class WindowLifecycle:
    """启动/停止/退出/持久化 — 生命周期职责单元。"""

    # ── 配置持久化 ──

    def _load_gui_config(self) -> None:
        """从持久化配置加载 GUI 状态。"""
        from gui.configs import load_config

        from gui.infer.state.bindings import params_from_dict

        config = load_config()
        saved_params = config.get("gui", {})

        state = params_from_dict(saved_params)
        # 原地更新 runtime_params 字段（不替换对象引用，engine/runner 自动生效）
        self.runtime_params.update_from(state)
        # apply_params 把控件值设置为 params 的值
        self.apply_gui_state(state)

        # 模型路径按钮：根据 win.model_path 更新显示文件名
        if self.model_path:
            from pathlib import Path

            self.model_path_btn.setText(Path(self.model_path).name)
            self.model_path_btn.setToolTip(self.model_path)

    def _save_gui_config(self) -> None:
        """保存当前 GUI 状态到持久化配置。"""
        from gui.configs import load_config, save_config

        from gui.infer.state.bindings import params_to_dict

        config = load_config()
        config["gui"] = params_to_dict(self.collect_gui_state())
        save_config(config)

    # ── 托盘退出与窗口关闭 ──

    def _tray_quit(self):
        """托盘退出：完整清理（停 timer/加载线程/离线/引擎/保存配置）后退出应用。"""
        self._timer.stop()

        if self._lt and self._lt.isRunning():
            self._lt.quit()
            self._lt.wait(2000)

        try:
            self._save_gui_config()
        except Exception as e:
            logger.error("保存配置失败：%s", e, exc_info=True)

        self.controller.stop()

        QApplication.instance().quit()

    def closeEvent(self, event):
        """点关闭按钮 → 隐藏到托盘继续运行（变声不中断），不退出。"""
        event.ignore()
        self.hide()
        if self.tray is not None:
            self.tray.notify_minimized()

    # ── 引擎预热 ──

    def _warmup_engine(self):
        """后台线程预热 engine — 首次构造会加载 torch 并做 CUDA 探测。"""
        import threading

        def _do():
            try:
                self.engine  # 触发惰性构造
                logger.info("引擎预热完成")
            except Exception as e:
                logger.warning("引擎预热失败（点开始时将再次尝试）", exc_info=True)
                # 无 GPU 等致命错误：主线程弹友好提示（后台线程不能直接操作 Qt）
                msg = str(e)
                QTimer.singleShot(0, lambda: self._show_error(f"引擎初始化失败: {msg}\n\n请确认已安装 NVIDIA 显卡驱动并支持 CUDA。"))

        threading.Thread(target=_do, daemon=True, name="engine-warmup").start()

    # ── 实时遥测刷新 ──

    def _update_timer(self):
        snap = self.controller.snapshot()

        if snap.running and snap.measure_ms > 0:
            self.delay_lbl.setText(f"延迟: {snap.measure_ms:.0f}ms")

            # 显示当前输入音高（音域映射之前的原始值）
            if snap.input_pitch > 0:
                self.pitch_lbl.setText(f"音高: {snap.input_pitch:.0f}Hz")
            else:
                self.pitch_lbl.setText("音高: -")

        # 错误计数：音频回调吞错（输出清零但引擎继续跑）的唯一可见信号
        if snap.error_count > 0:
            self.error_lbl.setText(f"错误: {snap.error_count}")
            self.error_lbl.show()
        else:
            self.error_lbl.hide()

        if self.tray is not None:
            self.tray.update_status()

    # ── 启动/停止 ──

    def _on_toggle_clicked(self):
        if self.engine.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        pth = self.model_path.strip()

        if not pth:
            self._show_warning("请先在参数调节中选择模型文件")
            return

        hubert = self.hubert_combo.currentText()

        self._apply_runtime_params()

        # 保存配置（在启动前保存当前设置）
        try:
            self._save_gui_config()
            logger.debug("配置已保存")
        except Exception as e:
            logger.warning("保存配置失败：%s", e)

        self._start_engine(pth, hubert)

    def _start_engine(self, pth, hubert):
        if self._loading:
            old = self._lt

            if old and old.isRunning():
                old.request_stop()
                if not old.wait(3000):
                    logger.warning("旧模型加载线程 3s 内未结束（load_model 阻塞中），断开信号后继续")

            for sig in (old.ok, old.err, old.finished):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass

            old.deleteLater()
            self._loading = False

        self._loading = True

        self.controller.begin_load()
        self._mark_loading()
        self._lt = LoadThread(self.engine, pth, hubert)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        self._lt.finished.connect(self._on_load_done)
        self._lt.start()

    def _on_load_done(self):
        self._loading = False
        self.controller.end_load()
        if self._lt:
            self._lt.deleteLater()
            self._lt = None

    def _on_loaded(self, sr):
        try:
            state = self.collect_gui_state()
            stats = self.controller.setup_engine(
                sr_mode=state.audio.sr_mode,
                input_device_idx=self.device_catalog.get_input_device_index(self.input_combo.currentIndex()),
                output_device_idx=self.device_catalog.get_output_device_index(self.output_combo.currentIndex()),
                output2_device_idx=self.device_catalog.get_output_device_index(self.output2_combo.currentIndex() - 1),
                block_time=state.buffer.block_time,
                crossfade_time=state.buffer.crossfade_time,
                extra_time=state.buffer.extra_time,
                enable_out2=state.audio.enable_out2,
            )
            self.sr_model_radio.setText(f"模型 {stats.sr_model}")
            self.sr_device_radio.setText(f"设备 {stats.sr_dev}")
            self._mark_running()
            if self.tray is not None:
                self.tray.update_status()
            self._save_gui_config()
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, e):
        self._reset_runtime_ui()
        self._show_error(str(e))

    def _on_runtime_error(self, message):
        # 在音频回调线程（PortAudio）调用：只发信号，所有 Qt 操作转发主线程执行
        self.runtime_error.emit(message)

    def _handle_runtime_error(self, message):
        # 主线程：流已在音频回调内通过 CallbackStop 安全停止；委托 controller 处理业务逻辑
        self.controller.handle_runtime_error(message)
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        self._show_error(f"实时推理错误: {message}")

    def _stop(self):
        if self._loading:
            if self._lt and self._lt.isRunning():
                self._lt.request_stop()
                self._lt.wait(3000)
            self._loading = False
            self.controller.end_load()
            self._reset_runtime_ui()
            if self.tray is not None:
                self.tray.update_status()
            return

        if not self.engine.running:
            return

        self.controller.stop_inference()
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("停止")

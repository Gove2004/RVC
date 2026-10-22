"""MainWindow 的控制器契约实现 — OfflineViewHost / DeviceViewHost 的控件机械操作。

controller 层禁止触碰控件；本模块把「语义」归结为「控件机械操作」。
独立成 mixin 只为文件组织：window.py 保留装配 / 信号连接 / 模式切换，
全部宿主契约方法集中在这里。按钮文案恒 2 字（换长度会撑动整列宽度，刻意设计）。
"""
from PySide6.QtWidgets import QMessageBox

from gui.state.bindings import OUTPUT2_NONE_TEXT


class HostContractMixin:
    """MainWindow 混入：离线转换与设备下拉框的宿主契约实现。"""

    # ── 离线转换 OfflineViewHost ──

    def offline_input_path(self) -> str:
        return self.offline_input_btn.path().strip()

    def offline_model_path(self) -> str:
        return self.model_path_btn.path().strip()

    def is_engine_loading(self) -> bool:
        """模型是否正在后台加载（供离线转换等需要互斥的路径判断）。

        不能只看 _loading 布尔量：停止路径取消加载时只是放弃结果，线程本身
        仍在 load_model 里（拿不到安全中断点），仍占着 engine._load_lock。
        以「线程实际是否在跑」为准，离线 worker 才不会静默卡在那把锁上。
        """
        if self._loading:
            return True
        lt = self._lt
        if lt is None:
            return False
        try:
            return lt.isRunning()
        except RuntimeError:
            return False  # C++ 对象已随 deleteLater 回收

    def offline_confirm_delete(self, path: str) -> bool:
        ans = QMessageBox.question(
            self, "删除输出文件",
            f"确定删除该输出文件吗？\n{path}\n\n文件将移入回收站。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return ans == QMessageBox.StandardButton.Yes

    def offline_enter_converting(self) -> None:
        self.offline_button.setText("取消")
        # 输入文件也锁住：输出路径由输入派生，转换途中改选输入会让「完成」
        # 之后的输出行/监听/删除全部指向另一个文件（用户听到或删掉的是旧产物）
        self.offline_input_btn.setEnabled(False)
        self.offline_preview_btn.setEnabled(False)
        self.offline_delete_btn.setEnabled(False)
        self.offline_status.setText("初始化...")
        self.offline_progress.setValue(0)

    def offline_enter_cancelling(self) -> None:
        self.offline_button.setEnabled(False)
        self.offline_status.setText("正在取消...")

    def offline_reset(self, status: str) -> None:
        self.offline_button.setEnabled(True)
        self.offline_button.setText("开始")
        self.offline_input_btn.setEnabled(True)
        self.offline_preview_btn.setEnabled(True)
        self.offline_delete_btn.setEnabled(True)
        self.offline_status.setText(status)

    def offline_set_progress(self, current: int, total: int) -> None:
        """只推进度条。状态行由 controller 经 offline_set_status 单独控制——
        取消时状态行要停在「正在取消...」，不能被进度覆盖。"""
        if total > 0:
            self.offline_progress.setValue(current)

    def offline_set_status(self, text: str) -> None:
        self.offline_status.setText(text)

    def offline_set_previewing(self, on: bool, name: str = "") -> None:
        self.offline_preview_btn.setText("停止" if on else "监听")
        if on:
            self.offline_status.setText(f"监听中: {name}")

    # ── 设备下拉框 DeviceViewHost ──

    def device_current_selections(self) -> tuple[str, str, str, str]:
        return (
            self.hostapi_combo.currentText(),
            self.input_combo.currentText(),
            self.output_combo.currentText(),
            self.output2_combo.currentText(),
        )

    def device_populate(self, hostapis, ins, outs, sel) -> None:
        """重填四个设备下拉框并恢复原选择。

        四个下拉框全部屏蔽信号后再重填：currentTextChanged 会触发
        _ha_changed 递归刷新（PortAudio 全量枚举跑两遍）；副输出重填期间的
        clear() 还会让 _on_output2_changed 把 enable_out2 瞬时写成 False
        （设备不在新列表时甚至会永久脱节）。派生字段在重填后统一同步一次。
        副输出首项恒为占位「不使用」（单源 bindings）。
        """
        ha, inp, outp, out2 = sel
        combos = (self.hostapi_combo, self.input_combo,
                  self.output_combo, self.output2_combo)
        for combo in combos:
            combo.blockSignals(True)
        try:
            self.hostapi_combo.clear()
            self.hostapi_combo.addItems(hostapis)
            self._select_combo(self.hostapi_combo, ha)

            self.input_combo.clear()
            self.input_combo.addItems(ins)
            self.output_combo.clear()
            self.output_combo.addItems(outs)
            self.output2_combo.clear()
            self.output2_combo.addItem(OUTPUT2_NONE_TEXT)
            self.output2_combo.addItems(outs)
            for combo, text in ((self.input_combo, inp),
                                (self.output_combo, outp),
                                (self.output2_combo, out2)):
                self._select_combo(combo, text)
            # 信号屏蔽期间不会触发，这里补一次派生同步（也保证恢复失败时
            # 派生字段如实反映「不使用」）
            self._on_output2_changed(self.output2_combo.currentText())
        finally:
            for combo in combos:
                combo.blockSignals(False)

    @staticmethod
    def _select_combo(combo, text: str) -> None:
        """若 text 仍在列表中则恢复选中，否则保持默认首项。"""
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

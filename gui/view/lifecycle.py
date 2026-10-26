"""推理 GUI 生命周期 — MainWindow 的生命周期职责单元。

职责：配置持久化、引擎预热、启动/停止（加载线程管理）、
运行时错误处理、托盘退出、关闭行为。UI 装配与信号绑定在 window.py。

Mixin 形式：方法挂在 MainWindow 上，通过 self 访问窗口的控件与 controller。
"""
import logging

from PySide6.QtWidgets import QApplication

from gui.view.widgets import LoadThread

logger = logging.getLogger(__name__)


class WindowLifecycle:
    """启动/停止/退出/持久化 — 生命周期职责单元。

    宿主契约：`gui.view.contracts.InferWindowHost`（MainWindow 结构化满足之；
    本混入不重复声明属性，contracts.py 为唯一事实源）。
    """

    # ── 配置持久化 ──

    def _load_gui_config(self) -> None:
        """从持久化配置加载 GUI 状态。"""
        from gui.configs import load_config

        from gui.state.bindings import params_from_dict

        config = load_config()
        saved_params = config.get("gui", {})

        state = params_from_dict(saved_params)
        # 原地更新 runtime_params 字段（不替换对象引用，engine/runner 自动生效）
        self.runtime_params.update_from(state)
        # apply_params 把控件值设置为 params 的值（模型路径按钮由 PATH 方式一并回填）
        self.apply_gui_state(state)

    def _save_gui_config(self) -> None:
        """保存当前 GUI 状态到持久化配置。"""
        from gui.configs import load_config, save_config

        from gui.state.bindings import params_to_dict

        config = load_config()
        config["gui"] = params_to_dict(self.collect_gui_state())
        save_config(config)

    # ── 托盘退出与窗口关闭 ──

    def _tray_quit(self):
        """托盘退出：完整清理（停 timer/加载线程/离线转换/引擎/保存配置）后退出应用。

        线程清理整体包在一个 try 里：此刻托盘图标已被 hide，任何异常冒出都会
        让后面的保存配置/停引擎/quit 全部跳过——用户看到的是「图标没了、
        窗口没了、进程还在」，只能去任务管理器。清理失败也要把剩下的事做完。
        """
        self._timer.stop()

        try:
            self._stop_background_threads()
        except Exception:
            logger.exception("退出前清理后台线程失败，继续退出")

        try:
            self._save_gui_config()
        except Exception as e:
            logger.error("保存配置失败：%s", e, exc_info=True)

        self.controller.stop()
        self.controller.stop_passthrough()

        QApplication.instance().quit()

    def _stop_background_threads(self) -> None:
        """退出路径专用：等三个后台 QThread 结束（唯一的允许等待之处）。"""
        # LoadThread.run() 内部没有事件循环，quit() 对它无效（只会留下残线程），
        # 必须先 request_stop() 让 run() 自己退出，再等它结束。
        lt = self._lt
        if lt is not None:
            try:
                if lt.isRunning():
                    lt.request_stop()
                    if not lt.wait(3000):
                        logger.warning("加载线程 3s 内未结束，继续退出")
            except RuntimeError:
                pass  # C++ 对象已回收

        # 离线转换 worker 是 QThread（非 daemon），且可能正卡在 CUDA 推理中。
        # 不处理会让进程挂在退出上（表现为「点了退出没反应」）。
        self.offline_conversion.shutdown()

        # 模型工具线程（合并 / 检视）同为 QThread，退出前等它结束；
        # 排队中的请求随退出丢弃（没人再消费队列，也不该在关窗后写模型文件）
        runner = self.tool_runner
        if runner is not None:
            try:
                if runner.isRunning() and not runner.wait(5000):
                    logger.warning("模型工具线程 5s 内未结束，继续退出")
            except RuntimeError:
                pass
        self.tool_queue.clear()

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
                # torch 此刻已就绪，把显卡列表投递回主线程回填下拉框
                # （信号跨线程，Qt 自动走队列投递，不违反「后台线程不碰 Qt」）
                self.gpu_list_ready.emit(self.controller.list_gpus())
                # 顺手预加载上次模型（同在后台线程）：点「变声」时跳过
                # torch.load + 搬 GPU（~1-2s），只剩 warmup + 开流
                self.controller.preload_last_model()
            except Exception as e:
                logger.warning("引擎预热失败（点开始时将再次尝试）", exc_info=True)
                # 无 GPU 等致命错误：经信号投递到主线程弹提示（后台线程不能
                # 直接操作 Qt；singleShot 在非 Qt 线程上回调永不执行）
                self.warmup_error.emit(
                    f"引擎初始化失败: {e}\n\n请确认已安装 NVIDIA 显卡驱动并支持 CUDA。"
                )

        threading.Thread(target=_do, daemon=True, name="engine-warmup").start()

    # ── 实时遥测刷新 ──

    def _update_timer(self):
        snap = self.controller.snapshot()

        if snap.running and snap.measure_ms > 0:
            self.delay_lbl.setText(f"延迟: {snap.measure_ms:.0f}ms")
            # 动态 tooltip：显示延迟分解（内容未变不重写——定时器每 tick 都会
            # 走到这里，重写 tooltip 会触发一次无效的 Qt 属性变更/重绘）
            tooltip = (
                f"算法延迟 = 2×采样长度 + HuBERT未来上下文 + 推理\n"
                f"  输入块收集: {snap.driver_in_ms:.0f}ms\n"
                f"  HuBERT未来上下文: {snap.lookahead_ms:.0f}ms\n"
                f"  推理耗时: {snap.infer_ms:.0f}ms（GPU 端到端）\n"
                f"  输出块缓冲: {snap.driver_out_ms:.0f}ms\n"
                f"  合计: {snap.measure_ms:.0f}ms\n"
                f"注：WASAPI共享模式驱动层缓冲（系统混音器+VB-Cable）未计入，\n"
                f"实际听感通常比显示值大 100-150ms。换 WASAPI独占/ASIO 可降低。"
            )
            if tooltip != self._last_delay_tooltip:
                self._last_delay_tooltip = tooltip
                self.delay_lbl.setToolTip(tooltip)

            # 显示当前输入音高（音域映射之前的原始值）。文本未变不重写——
            # QLabel.setText 即使内容相同也会触发一次 repaint，按 tick 重写无谓。
            pitch_text = f"音高: {snap.input_pitch:.0f}Hz" if snap.input_pitch > 0 else "音高: -"
            if pitch_text != self._last_pitch_text:
                self._last_pitch_text = pitch_text
                self.pitch_lbl.setText(pitch_text)

        # 错误计数：音频回调吞错（输出清零但引擎继续跑）的唯一可见信号
        if snap.error_count > 0:
            if snap.error_count != self._last_error_count:
                self._last_error_count = snap.error_count
                self.error_lbl.setText(f"错误: {snap.error_count}")
                self.error_lbl.show()
        elif self._last_error_count != 0:
            self._last_error_count = 0
            self.error_lbl.hide()

        # 主循环已持有快照，传给托盘复用，避免同 tick 内二次 snapshot()
        if self.tray is not None:
            self.tray.update_status(snap)

    # ── 三档运行态（原声直通 / 停止 / 变声）──

    def _on_mode_changed(self, mode: str):
        """ModeToggle 档位分发：控件只报意图，档位执行与 UI 收敛在这里。"""
        {"passthrough": self._on_mode_passthrough,
         "stop": self._on_mode_stop,
         "convert": self._on_mode_convert}[mode]()

    def _on_mode_passthrough(self):
        """原声档：麦克风原样转发到主输出（不经模型）。

        从停止档进入；从变声档点击则先停引擎再开直通（同一路径逐级让位）。
        """
        if self._loading:
            self._refresh_mode()
            return
        if self.controller.is_running:
            self._stop()
        if self.controller.passthrough_running:
            return
        try:
            in_idx = self.device_catalog.get_input_device_index(
                self.input_combo.currentIndex()
            )
            out_idx = self.device_catalog.get_output_device_index(
                self.output_combo.currentIndex()
            )
            self.controller.start_passthrough(in_idx, out_idx)
        except Exception as e:
            self._refresh_mode()
            self.show_error(f"原声直通启动失败：{e}")
            return
        self._refresh_mode()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("原声直通已开启")

    def _on_mode_convert(self):
        """变声档：停直通（若有）→ 校验模型 → 加载引擎并启动推理。"""
        if self._loading or self.controller.is_running:
            # 加载中点「变声」：ModeToggle 已先勾上目标档，这里拒绝执行须回弹，
            # 否则 UI 显示「变声」勾选而实际仍处加载态（要等 _on_load_done 才修正）
            self._refresh_mode()
            return
        # 直通先让位：同一对输入/输出设备（_on_loaded setup 前还会再关一次，双保险）
        self.controller.stop_passthrough()
        self._start()

    def _on_mode_stop(self):
        """停止档：停掉当前运行态（加载中→取消加载；变声→停引擎；原声→停直通）。"""
        if self._loading:
            self._cancel_load()
            self._reset_runtime_ui()
        elif self.controller.is_running:
            self._stop()
        else:
            self.controller.stop_passthrough()
            self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("停止")

    # ── 启动/停止 ──

    def _start(self):
        pth = self.model_path_btn.path().strip()

        if not pth:
            self.show_warning("请先在「音色」中选择模型文件")
            self._refresh_mode()  # 拒绝启动：toggle 勾选态回弹到实际状态
            return

        # 离线/实时共用同一 engine/runner（process_file 会替换 _runner），
        # 离线转换进行中启动实时会并发 load_model 竞态——双向互斥
        if self.offline_conversion.is_converting:
            self.show_warning("离线转换进行中，请等待完成或取消后再启动实时")
            self._refresh_mode()
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
        """起一个后台加载线程（调用方已保证此刻没有别的加载在跑）。

        唯一调用方 _on_mode_convert 在入口就挡掉了加载中的情况，所以这里不需要
        额外的清理。真正可能并存的旧线程来自「取消加载后线程仍在跑」，由
        _on_load_done(lt) 认代回收。
        """
        self._loading = True

        self._mark_loading()
        self._lt = LoadThread(self.engine, pth, hubert)
        self._lt.ok.connect(self._on_loaded)
        self._lt.err.connect(self._on_err)
        # 线程实例绑进连接：迟到的那一代线程只回收自己，绝不 deleteLater
        # 当前正在跑的新线程（否则 Qt 报 "Destroyed while thread is still running"）
        lt = self._lt
        self._lt.finished.connect(lambda: self._on_load_done(lt))
        self._lt.start()

    def _on_load_done(self, lt=None):
        """线程结束收尾（lt = 发出 finished 的那一代线程）。"""
        if lt is not None and lt is not self._lt:
            # 已被取消并让位给新加载的旧线程：只回收它自己，不动 UI 状态
            logger.info("旧加载线程结束（已被新一代取代），仅回收")
            lt.deleteLater()
            return
        self._loading = False
        if self._lt is not None:
            self._lt.deleteLater()
            self._lt = None
        # 加载结束（成功→引擎已运行；失败→引擎未运行）：按实际状态收敛三档 UI
        self._refresh_mode()

    def _on_loaded(self, sr):
        if not self._loading:
            # 停止路径已取消这次加载（信号可能在断开之前就已入队）：迟到的
            # 加载结果绝不能把引擎拉起来——「点了停止却开始变声」是本应用最忌讳的故障。
            logger.info("忽略已取消的加载结果（sr=%s）", sr)
            return
        try:
            # 直通流先让位：它占着同一对设备，推理流起来前必须关（否则双重出声）
            self.controller.stop_passthrough()
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
        except Exception as e:
            self._on_err(str(e))
            return

        self.sr_model_radio.setText(f"模型 {stats.model_sr}")
        self.sr_device_radio.setText(f"设备 {stats.device_sr}")
        self._mark_running()
        if self.tray is not None:
            self.tray.update_status()

        # 存配置单独 try：磁盘满/只读目录/文件被占用都属于「引擎已正常起流、
        # 只是记不住偏好」，与推理失败不是一类事。若与 setup_engine 共用一个
        # try，写盘失败会走 _on_err 弹出「实时推理错误」并把遥测面板复位成
        # 停止态——用户看到报错却仍在变声，之后整段会话不再刷新遥测。
        try:
            self._save_gui_config()
        except Exception as e:
            logger.warning("保存配置失败（不影响本次变声）：%s", e)

    def _on_err(self, e):
        if not self._loading:
            # 同 _on_loaded：已取消的加载不该再弹错误框打扰用户
            logger.info("忽略已取消加载的错误：%s", e)
            return
        # 不变量：出错一律收敛到停止档，严禁回落原声——目标软件固定录
        # 虚拟声卡，错误后突然传出原声会露馅。直通在这里显式关掉
        # （即使理论上没开），_refresh_mode 只读状态、绝不自动开直通。
        # controller.stop() 兜住「流已起来、后续步骤才失败」的半启动状态：
        # 只复位 UI 不停引擎的话，界面显示停止而麦克风仍在往虚拟声卡送声音。
        self.controller.stop_passthrough()
        self.controller.stop()
        self._reset_runtime_ui()
        self.show_error(str(e))

    def _on_runtime_error(self, message):
        # 在音频回调线程（PortAudio）调用：只发信号，所有 Qt 操作转发主线程执行
        self.runtime_error.emit(message)

    def _handle_runtime_error(self, message):
        # 主线程：流已在音频回调内通过 CallbackStop 安全停止；委托 controller 处理业务逻辑
        self.controller.handle_runtime_error(message)
        # 不变量：运行时错误 → 停止档，绝不回落原声（露馅风险）。
        # 显式关直通做双保险，UI 由 _reset_runtime_ui 按流状态收敛为停止。
        self.controller.stop_passthrough()
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        self.show_error(f"实时推理错误: {message}")

    def _cancel_load(self) -> None:
        """取消正在进行的模型加载（停止路径调用，避免各写一遍漏掉断连）。

        **必须先断开 ok/err**：LoadThread.run() 只在入口检查一次停止标志，
        加载一旦开始就一定会 emit 结果；这些信号若仍连着，_on_loaded 会把
        引擎启动起来——用户点了「停止」却开始变声。

        **不等待线程**：在 GUI 线程 wait 会把界面按到加载结束（大模型冷盘/
        机械盘会真的卡满 3 秒，Windows 判定「无响应」），而且超时后线程仍在跑、
        _loading 已置 False，反而让 is_engine_loading 守卫失效。等待想保证的
        两件事各有明确机制：
        - 不误起引擎：断开 ok/err + _on_loaded/_on_err 的「已取消」守卫；
        - 不误删线程：_on_load_done(lt) 按线程实例认代，迟到的旧线程只回收自己。
        线程照常跑完（load_model 无法安全中断），由 finished 自行回收。
        """
        lt = self._lt
        self._loading = False
        if lt is None:
            return
        for sig in (lt.ok, lt.err):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        if lt.isRunning():
            lt.request_stop()  # 仅在 run() 下一次检查点生效；真要等就等下一块
        else:
            lt.deleteLater()
            if self._lt is lt:
                self._lt = None

    def _stop(self):
        """停引擎（变声档 → 停止档）。取消加载与停直通走 _on_mode_stop。"""
        if self._loading:
            self._cancel_load()
            self._reset_runtime_ui()
            if self.tray is not None:
                self.tray.update_status()
            return

        if not self.controller.is_running:
            return

        self.controller.stop()
        self._reset_runtime_ui()
        if self.tray is not None:
            self.tray.update_status()
        logger.info("停止")

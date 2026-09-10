"""全局主题 — QPalette + 全局 QSS 样式表

把 app.py 里的 _set_dark 函数移到这里，所有颜色从 Colors 类获取，
避免颜色硬编码分散在多处。

用法:
    from gui.styles import apply_theme
    apply_theme(app)
"""
from PySide6.QtGui import QPalette, QColor

from gui.styles.colors import Colors
from gui.styles.layout import Layout


def _build_palette() -> QPalette:
    """构建 Windows 11 深色模式 QPalette。"""
    pal = QPalette()

    # 背景层
    pal.setColor(QPalette.Window, QColor(Colors.WINDOW_BG))
    pal.setColor(QPalette.Base, QColor(Colors.INPUT_BG))
    pal.setColor(QPalette.AlternateBase, QColor(Colors.ALTERNATE_BG))
    pal.setColor(QPalette.ToolTipBase, QColor(Colors.TOOLTIP_BG))
    pal.setColor(QPalette.ToolTipText, QColor(Colors.TEXT_WHITE))

    # 文字
    pal.setColor(QPalette.WindowText, QColor(Colors.TEXT_WHITE))
    pal.setColor(QPalette.Text, QColor(Colors.TEXT_INPUT))
    pal.setColor(QPalette.ButtonText, QColor(Colors.TEXT_WHITE))
    pal.setColor(QPalette.BrightText, QColor(Colors.TEXT_WHITE))

    # 控件
    pal.setColor(QPalette.Button, QColor(Colors.SECONDARY))

    # 强调色（Windows Blue）
    pal.setColor(QPalette.Highlight, QColor(Colors.PRIMARY))
    pal.setColor(QPalette.HighlightedText, QColor(Colors.TEXT_WHITE))
    pal.setColor(QPalette.Link, QColor(Colors.LINK))

    # 禁用态
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(Colors.DISABLED_TEXT))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(Colors.DISABLED_TEXT))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(Colors.DISABLED_TEXT))
    pal.setColor(QPalette.Disabled, QPalette.Button, QColor(Colors.DISABLED_BG))
    pal.setColor(QPalette.Disabled, QPalette.Base, QColor(Colors.INPUT_BG))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, QColor(Colors.DISABLED_HIGHLIGHT))

    return pal


def _build_global_stylesheet() -> str:
    """构建全局 QSS 样式表（Windows 11 风格控件）。"""
    r = Layout.RADIUS_NORMAL
    return f"""
        /* GroupBox */
        QGroupBox {{
            font-weight: bold;
            margin-top: 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 3px;
        }}

        /* Slider */
        QSlider {{
            min-height: 18px;
        }}
        QSlider::groove:horizontal {{
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 10px;
            height: 10px;
            margin: -3px 0;
            background-color: {Colors.PRIMARY};
            border: 1px solid {Colors.INPUT_BG};
            border-radius: 2px;
        }}

        /* 输入控件 */
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {{
            border: 1px solid {Colors.BORDER};
            border-radius: {r}px;
            background-color: {Colors.INPUT_BG};
            padding: 3px 6px;
            selection-background-color: {Colors.PRIMARY};
        }}
        QComboBox {{
            border: 1px solid {Colors.BORDER};
            border-radius: {r}px;
            background-color: {Colors.SECONDARY};
            color: {Colors.TEXT_PRIMARY};
            padding: 3px 6px;
        }}
        QComboBox:hover {{
            border-color: {Colors.PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {Colors.SECONDARY};
            color: {Colors.TEXT_PRIMARY};
            selection-background-color: {Colors.PRIMARY};
            outline: none;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border-color: {Colors.PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 22px;
        }}
        QComboBox QAbstractItemView {{
            border: 1px solid {Colors.BORDER};
            background-color: {Colors.SECONDARY};
            selection-background-color: {Colors.PRIMARY};
            outline: none;
        }}

        /* TabWidget */
        QTabWidget::pane {{
            border: 1px solid {Colors.BORDER};
            top: -1px;
        }}
        QTabBar::tab {{
            background-color: {Colors.WINDOW_BG};
            padding: 6px 14px;
            border: 1px solid {Colors.BORDER};
            border-bottom: none;
        }}
        QTabBar::tab:selected {{
            background-color: {Colors.SECONDARY};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {Colors.ALTERNATE_BG};
        }}

        /* ScrollBar */
        QScrollBar:vertical {{
            background: {Colors.WINDOW_BG};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {Colors.BORDER};
            border-radius: 5px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: #4a4a4a;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {Colors.WINDOW_BG};
            height: 10px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {Colors.BORDER};
            border-radius: 5px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: #4a4a4a;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        /* Menu */
        QMenu {{
            background-color: {Colors.SECONDARY};
            border: 1px solid {Colors.BORDER};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 5px 20px;
            border-radius: {r}px;
        }}
        QMenu::item:selected {{
            background-color: {Colors.PRIMARY};
        }}
        QMenu::separator {{
            height: 1px;
            background: {Colors.BORDER};
            margin: 4px 8px;
        }}

        /* CheckBox / RadioButton */
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            border: 1px solid {Colors.CHECKBOX_BORDER};
            border-radius: 3px;
            background: {Colors.INPUT_BG};
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background: {Colors.PRIMARY};
            border-color: {Colors.PRIMARY};
        }}

        /* ProgressBar */
        QProgressBar {{
            border: 1px solid {Colors.BORDER};
            border-radius: {r}px;
            background-color: {Colors.INPUT_BG};
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {Colors.PRIMARY};
            border-radius: 3px;
        }}
    """


def apply_theme(app) -> None:
    """应用全局主题（QPalette + QSS）。

    Args:
        app: QApplication 实例
    """
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(_build_global_stylesheet())

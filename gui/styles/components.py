"""按钮样式 — 统一的按钮 QSS 样式生成器"""
from gui.styles.colors import Colors
from gui.styles.layout import Layout


class ButtonStyles:
    """统一的按钮样式"""

    @staticmethod
    def _button_qss(bg: str, hover: str, disabled: str, *,
                    padding: str = "5px 12px",
                    font_weight: str = "bold") -> str:
        """主/次/危险按钮共用的 QSS 模板（仅色值与字重不同）。

        刻意不接受 width/height：按钮尺寸由布局（`_pin_button_width` 等）决定，
        在样式里写死宽高会让布局约束失效，二者迟早打架。
        """
        return f"""
            QPushButton {{
                background: {bg};
                color: {Colors.TEXT_WHITE};
                font-weight: {font_weight};
                padding: {padding};
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QPushButton:disabled {{
                background: {disabled};
                color: {Colors.MUTED_TEXT};
            }}
        """

    @classmethod
    def primary(cls) -> str:
        """主要操作按钮（绿色）"""
        return cls._button_qss(Colors.PRIMARY, Colors.PRIMARY_HOVER, Colors.PRIMARY_DISABLED)

    @classmethod
    def secondary(cls) -> str:
        """次要操作按钮（蓝色）"""
        return cls._button_qss(Colors.SECONDARY, Colors.SECONDARY_HOVER, Colors.SECONDARY_DISABLED)

    @classmethod
    def danger(cls) -> str:
        """危险操作按钮（红色）"""
        return cls._button_qss(Colors.DANGER, Colors.DANGER_HOVER, Colors.DANGER_DISABLED)

    @staticmethod
    def path_button() -> str:
        """路径选择按钮（黑色背景，文字左对齐，用于模型路径、输入文件等）"""
        return f"""
            QPushButton {{
                background-color: {Colors.INPUT_BG};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: {Layout.RADIUS_NORMAL}px;
                padding: 3px 8px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {Colors.ALTERNATE_BG};
                border-color: {Colors.PRIMARY};
            }}
        """

"""按钮样式 — 统一的按钮 QSS 样式生成器"""
from gui.styles.colors import Colors
from gui.styles.layout import Layout


class ButtonStyles:
    """统一的按钮样式"""

    @staticmethod
    def _button_qss(bg: str, hover: str, disabled: str, *,
                    size: str = "", padding: str = "5px 12px",
                    font_weight: str = "bold") -> str:
        """主/次/危险按钮共用的 QSS 模板（仅色值与字重不同）。"""
        return f"""
            QPushButton {{
                {size}
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
    def primary(cls, width=None, height=None):
        """主要操作按钮（绿色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return cls._button_qss(Colors.PRIMARY, Colors.PRIMARY_HOVER, Colors.PRIMARY_DISABLED, size=size)

    @classmethod
    def secondary(cls, width=None, height=None):
        """次要操作按钮（蓝色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return cls._button_qss(Colors.SECONDARY, Colors.SECONDARY_HOVER, Colors.SECONDARY_DISABLED, size=size)

    @classmethod
    def danger(cls, width=None, height=None):
        """危险操作按钮（红色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return cls._button_qss(Colors.DANGER, Colors.DANGER_HOVER, Colors.DANGER_DISABLED, size=size)

    @staticmethod
    def mode_active(bg: str) -> str:
        """三档运行态按钮的当前档样式：实心亮色（状态色即身份色）。"""
        return f"""
            QPushButton {{
                background: {bg};
                color: {Colors.TEXT_WHITE};
                font-weight: bold;
                padding: 5px 10px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {bg};
            }}
        """

    @staticmethod
    def mode_idle(color: str) -> str:
        """三档运行态按钮的非当前档样式：暗底 + 状态色描边与文字。"""
        return f"""
            QPushButton {{
                background: {Colors.SECONDARY};
                color: {color};
                font-weight: bold;
                padding: 5px 10px;
                border: 1px solid {color};
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {Colors.SECONDARY_HOVER};
            }}
            QPushButton:disabled {{
                background: {Colors.SECONDARY_DISABLED};
                color: {Colors.MUTED_TEXT};
                border-color: {Colors.BORDER};
            }}
        """

    @staticmethod
    def small(color="secondary"):
        """小按钮（查看、修改原名、刷新等）"""
        colors = {
            "primary": (Colors.PRIMARY, Colors.PRIMARY_HOVER),
            "secondary": (Colors.SECONDARY, Colors.SECONDARY_HOVER),
            "danger": (Colors.DANGER, Colors.DANGER_HOVER),
        }
        bg, hover = colors.get(color, colors["secondary"])

        return f"""
            QPushButton {{
                background: {bg};
                color: {Colors.TEXT_WHITE};
                font-weight: normal;
                padding: 0px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
            QPushButton:disabled {{
                background: {Colors.MUTED_BG};
                color: {Colors.MUTED_TEXT};
            }}
        """


    @staticmethod
    def path_button():
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

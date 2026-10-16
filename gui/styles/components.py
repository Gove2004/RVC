"""按钮样式 — 统一的按钮 QSS 样式生成器"""
from gui.styles.colors import Colors
from gui.styles.layout import Layout


class ButtonStyles:
    """统一的按钮样式"""

    @staticmethod
    def primary(width=None, height=None):
        """主要操作按钮（绿色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return f"""
            QPushButton {{
                {size}
                background: {Colors.PRIMARY};
                color: {Colors.TEXT_WHITE};
                font-weight: bold;
                padding: 5px 12px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {Colors.PRIMARY_HOVER};
            }}
            QPushButton:disabled {{
                background: {Colors.PRIMARY_DISABLED};
                color: {Colors.MUTED_TEXT};
            }}
        """

    @staticmethod
    def secondary(width=None, height=None):
        """次要操作按钮（蓝色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return f"""
            QPushButton {{
                {size}
                background: {Colors.SECONDARY};
                color: {Colors.TEXT_WHITE};
                font-weight: bold;
                padding: 5px 12px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {Colors.SECONDARY_HOVER};
            }}
            QPushButton:disabled {{
                background: {Colors.SECONDARY_DISABLED};
                color: {Colors.MUTED_TEXT};
            }}
        """

    @staticmethod
    def danger(width=None, height=None):
        """危险操作按钮（红色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return f"""
            QPushButton {{
                {size}
                background: {Colors.DANGER};
                color: {Colors.TEXT_WHITE};
                font-weight: bold;
                padding: 5px 12px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
            QPushButton:hover {{
                background: {Colors.DANGER_HOVER};
            }}
            QPushButton:disabled {{
                background: {Colors.DANGER_DISABLED};
                color: {Colors.MUTED_TEXT};
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

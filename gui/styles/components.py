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
    def muted(width=None, height=None):
        """静音/禁用按钮（灰色）"""
        size = f"width:{width}px;" if width else ""
        size += f"height:{height}px;" if height else ""
        return f"""
            QPushButton {{
                {size}
                background: {Colors.MUTED_BG};
                color: {Colors.MUTED_TEXT};
                font-weight: bold;
                padding: 5px 12px;
                border: none;
                border-radius: {Layout.RADIUS_NORMAL}px;
            }}
        """

    @staticmethod
    def small(color="secondary"):
        """小按钮（浏览、刷新等）"""
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
                font-weight: bold;
                padding: 4px 8px;
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


class LabelStyles:
    """统一的标签样式"""

    @staticmethod
    def bold():
        """粗体标签"""
        return "font-weight: bold;"

    @staticmethod
    def status(status="default"):
        """状态标签"""
        colors = {
            "success": Colors.SUCCESS,
            "info": Colors.INFO,
            "warning": Colors.WARNING,
            "error": Colors.ERROR,
            "default": Colors.TEXT_PRIMARY,
        }
        color = colors.get(status, colors["default"])
        return f"color: {color}; font-weight: bold;"


class CardStyles:
    """统一的卡片样式"""

    @staticmethod
    def default():
        """默认卡片"""
        return f"""
            border: 1px solid {Colors.BORDER};
            border-radius: {Layout.RADIUS_NORMAL}px;
            margin: 1px;
        """

    @staticmethod
    def active(type="info"):
        """激活状态卡片"""
        colors = {
            "success": (Colors.SUCCESS, Colors.SUCCESS_BG),
            "info": (Colors.INFO, Colors.INFO_BG),
        }
        border, bg = colors.get(type, colors["info"])

        return f"""
            border: 1px solid {border};
            border-radius: {Layout.RADIUS_NORMAL}px;
            margin: 1px;
            background: {bg};
        """


class MiscStyles:
    """其他控件的统一样式"""

    @staticmethod
    def divider():
        """分隔线"""
        return f"color: {Colors.DIVIDER};"

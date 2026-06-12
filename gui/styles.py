"""RVC UI 统一样式系统

本模块定义了整个应用的统一视觉风格，包括：
- 颜色系统
- 按钮样式
- 布局参数
- 状态指示
"""

# ============================================================================
# 颜色系统
# ============================================================================

class Colors:
    """统一的颜色常量"""

    # 主题色
    PRIMARY = "#28a745"           # 绿色 - 主要操作
    PRIMARY_HOVER = "#218838"
    PRIMARY_DISABLED = "#555"

    SECONDARY = "#3b82f6"         # 蓝色 - 次要操作（统一使用此蓝）
    SECONDARY_HOVER = "#2b6cd9"
    SECONDARY_DISABLED = "#555"

    DANGER = "#dc3545"            # 红色 - 危险操作
    DANGER_HOVER = "#c82333"
    DANGER_DISABLED = "#555"

    # 中性色
    MUTED_BG = "#555"             # 禁用/静音背景
    MUTED_TEXT = "#888"           # 禁用/静音文字

    # 边框和分隔
    BORDER = "#444"
    DIVIDER = "#333"

    # 状态色
    SUCCESS = "#28a745"           # 成功
    INFO = "#3b82f6"              # 信息/进行中
    WARNING = "#f39c12"           # 警告
    ERROR = "#dc3545"             # 错误

    # 背景色（半透明用于卡片高亮）
    SUCCESS_BG = "rgba(40, 167, 69, 0.06)"
    INFO_BG = "rgba(59, 130, 246, 0.06)"

    # 文字色
    TEXT_PRIMARY = "#dcdcdc"      # 主要文字
    TEXT_SECONDARY = "#999"       # 次要文字
    TEXT_WHITE = "white"


# ============================================================================
# 布局参数
# ============================================================================

class Layout:
    """统一的布局参数"""

    # 边距
    WINDOW_MARGIN = 8             # 窗口边距
    TAB_MARGIN = 8                # Tab 边距
    CARD_MARGIN = 6               # 卡片边距

    # 间距
    SPACING_LARGE = 8             # 大间距（区块之间）
    SPACING_NORMAL = 6            # 标准间距（组件之间）
    SPACING_SMALL = 4             # 小间距（紧凑布局）

    # 按钮尺寸
    BTN_HEIGHT_NORMAL = 28        # 标准按钮高度
    BTN_HEIGHT_SMALL = 24         # 小按钮高度
    BTN_WIDTH_ICON = 28           # 图标按钮宽度
    BTN_WIDTH_SMALL = 32          # 小按钮宽度（浏览等）
    BTN_WIDTH_NORMAL = 60         # 标准按钮宽度（开始/停止）

    # 圆角
    RADIUS_NORMAL = 3             # 标准圆角
    RADIUS_LARGE = 4              # 大圆角


# ============================================================================
# 按钮样式
# ============================================================================

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


# ============================================================================
# 标签样式
# ============================================================================

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


# ============================================================================
# 卡片样式
# ============================================================================

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


# ============================================================================
# 其他控件样式
# ============================================================================

class MiscStyles:
    """其他控件的统一样式"""

    @staticmethod
    def divider():
        """分隔线"""
        return f"color: {Colors.DIVIDER};"

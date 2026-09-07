"""颜色系统 — Windows 11 深色模式（夜晚）配色"""


class Colors:
    """统一的颜色常量 - Windows 11 Dark Mode"""

    # 主题色 - Windows 强调色蓝
    PRIMARY = "#0078D4"           # Windows Blue - 主要操作
    PRIMARY_HOVER = "#1a86d9"
    PRIMARY_DISABLED = "#3a6a9a"

    SECONDARY = "#2a2a2a"         # 深灰 - 次要操作
    SECONDARY_HOVER = "#333333"
    SECONDARY_DISABLED = "#202020"

    DANGER = "#c42b1c"            # Windows 红 - 危险操作
    DANGER_HOVER = "#d13424"
    DANGER_DISABLED = "#7a2a20"

    # 中性色
    MUTED_BG = "#202020"          # 禁用/静音背景
    MUTED_TEXT = "#888888"        # 禁用/静音文字

    # 边框和分隔
    BORDER = "#3a3a3a"
    DIVIDER = "#2a2a2a"

    # 状态色 - Windows 系统色
    SUCCESS = "#107c10"           # Windows 绿
    INFO = "#0078D4"              # Windows 蓝
    WARNING = "#ff8c00"           # Windows 橙
    ERROR = "#c42b1c"             # Windows 红

    # 背景色（半透明用于卡片高亮）
    SUCCESS_BG = "rgba(16, 124, 16, 0.08)"
    INFO_BG = "rgba(0, 120, 212, 0.08)"

    # 背景色
    WINDOW_BG = "#202020"          # 窗口背景
    INPUT_BG = "#1a1a1a"           # 输入框/文本区背景
    ALTERNATE_BG = "#252525"       # 交替行背景
    TOOLTIP_BG = "#2a2a2a"         # 工具提示背景

    # 文字色
    TEXT_PRIMARY = "#ffffff"        # 主要文字
    TEXT_SECONDARY = "#c5c5c5"      # 次要文字
    TEXT_INPUT = "#e5e5e5"          # 输入框文字
    TEXT_WHITE = "white"
    LINK = "#4cc2ff"                # 链接色

    # 禁用态
    DISABLED_TEXT = "#6d6d6d"       # 禁用文字
    DISABLED_BG = "#323232"         # 禁用按钮背景
    DISABLED_HIGHLIGHT = "#3c3c3c"  # 禁用高亮

    # 控件边框
    CHECKBOX_BORDER = "#555555"     # 复选框/单选框边框

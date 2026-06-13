"""颜色系统 — 统一的颜色常量"""


class Colors:
    """统一的颜色常量 - 简洁灰色系"""

    # 主题色 - 统一灰色系
    PRIMARY = "#5a5a5a"           # 深灰 - 主要操作
    PRIMARY_HOVER = "#6a6a6a"
    PRIMARY_DISABLED = "#3a3a3a"

    SECONDARY = "#4a4a4a"         # 中灰 - 次要操作
    SECONDARY_HOVER = "#5a5a5a"
    SECONDARY_DISABLED = "#3a3a3a"

    DANGER = "#6a6a6a"            # 浅灰 - 危险操作（不突出）
    DANGER_HOVER = "#7a7a7a"
    DANGER_DISABLED = "#3a3a3a"

    # 中性色
    MUTED_BG = "#3a3a3a"          # 禁用/静音背景
    MUTED_TEXT = "#888"           # 禁用/静音文字

    # 边框和分隔
    BORDER = "#444"
    DIVIDER = "#333"

    # 状态色 - 也用灰色
    SUCCESS = "#5a5a5a"           # 成功
    INFO = "#4a4a4a"              # 信息/进行中
    WARNING = "#6a6a6a"           # 警告
    ERROR = "#6a6a6a"             # 错误

    # 背景色（半透明用于卡片高亮）
    SUCCESS_BG = "rgba(90, 90, 90, 0.06)"
    INFO_BG = "rgba(74, 74, 74, 0.06)"

    # 文字色
    TEXT_PRIMARY = "#dcdcdc"      # 主要文字
    TEXT_SECONDARY = "#999"       # 次要文字
    TEXT_WHITE = "white"

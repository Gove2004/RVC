"""颜色系统 — 统一的颜色常量"""


class Colors:
    """统一的颜色常量 - 简洁灰色系（高对比度）"""

    # 主题色 - 灰色系，与深色背景有足够对比
    PRIMARY = "#8a8a8a"           # 浅灰 - 主要操作（明显高于背景）
    PRIMARY_HOVER = "#9a9a9a"
    PRIMARY_DISABLED = "#3a3a3a"

    SECONDARY = "#7a7a7a"         # 中浅灰 - 次要操作
    SECONDARY_HOVER = "#8a8a8a"
    SECONDARY_DISABLED = "#3a3a3a"

    DANGER = "#6a6a6a"            # 中灰 - 危险操作
    DANGER_HOVER = "#7a7a7a"
    DANGER_DISABLED = "#3a3a3a"

    # 中性色
    MUTED_BG = "#3a3a3a"          # 禁用/静音背景（深灰）
    MUTED_TEXT = "#888"           # 禁用/静音文字

    # 边框和分隔
    BORDER = "#444"
    DIVIDER = "#333"

    # 状态色 - 也用灰色
    SUCCESS = "#8a8a8a"           # 成功
    INFO = "#7a7a7a"              # 信息/进行中
    WARNING = "#6a6a6a"           # 警告
    ERROR = "#6a6a6a"             # 错误

    # 背景色（半透明用于卡片高亮）
    SUCCESS_BG = "rgba(138, 138, 138, 0.06)"
    INFO_BG = "rgba(122, 122, 122, 0.06)"

    # 文字色
    TEXT_PRIMARY = "#dcdcdc"      # 主要文字
    TEXT_SECONDARY = "#999"       # 次要文字
    TEXT_WHITE = "white"

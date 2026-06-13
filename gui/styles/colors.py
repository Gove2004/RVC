"""颜色系统 — 统一的颜色常量"""


class Colors:
    """统一的颜色常量"""

    # 主题色
    PRIMARY = "#28a745"           # 绿色 - 主要操作
    PRIMARY_HOVER = "#218838"
    PRIMARY_DISABLED = "#555"

    SECONDARY = "#3b82f6"         # 蓝色 - 次要操作
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

"""颜色系统 — 统一的颜色常量"""


class Colors:
    """统一的颜色常量"""

    # 主题色 - 更柔和的现代配色
    PRIMARY = "#10b981"           # 翠绿色 - 主要操作（原 #28a745 太鲜艳）
    PRIMARY_HOVER = "#059669"
    PRIMARY_DISABLED = "#555"

    SECONDARY = "#6366f1"         # 靛蓝色 - 次要操作（原 #3b82f6 太亮）
    SECONDARY_HOVER = "#4f46e5"
    SECONDARY_DISABLED = "#555"

    DANGER = "#ef4444"            # 柔和红色 - 危险操作（原 #dc3545 太刺眼）
    DANGER_HOVER = "#dc2626"
    DANGER_DISABLED = "#555"

    # 中性色
    MUTED_BG = "#555"             # 禁用/静音背景
    MUTED_TEXT = "#888"           # 禁用/静音文字

    # 边框和分隔
    BORDER = "#444"
    DIVIDER = "#333"

    # 状态色
    SUCCESS = "#10b981"           # 成功
    INFO = "#6366f1"              # 信息/进行中
    WARNING = "#f59e0b"           # 警告（原 #f39c12 稍微调整）
    ERROR = "#ef4444"             # 错误

    # 背景色（半透明用于卡片高亮）
    SUCCESS_BG = "rgba(16, 185, 129, 0.06)"
    INFO_BG = "rgba(99, 102, 241, 0.06)"

    # 文字色
    TEXT_PRIMARY = "#dcdcdc"      # 主要文字
    TEXT_SECONDARY = "#999"       # 次要文字
    TEXT_WHITE = "white"

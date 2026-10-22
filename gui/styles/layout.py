"""布局参数 — 统一的尺寸和间距常量"""


class Layout:
    """统一的布局参数（只保留实际使用的常量，需要时再加回）"""

    # 间距
    SPACING_SMALL = 4             # 小间距（紧凑布局）

    # 按钮尺寸
    BTN_HEIGHT_NORMAL = 28        # 标准按钮高度
    BTN_WIDTH_NORMAL = 60         # 标准按钮宽度（开始/停止）

    # 圆角（Windows 11 风格）
    RADIUS_NORMAL = 4             # 标准圆角

    # 滑块控件高度（QSS 与 setFixedHeight 单源——改一处即全局生效）
    SLIDER_HEIGHT = 18

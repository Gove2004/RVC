"""布局参数 — 统一的尺寸和间距常量"""


class Layout:
    """统一的布局参数（只保留实际使用的常量，需要时再加回）"""

    # 间距
    SPACING_SMALL = 4             # 小间距（紧凑布局）

    # 按钮尺寸
    BTN_HEIGHT_NORMAL = 28        # 标准按钮高度
    # 按钮宽度不设常量：离线页按文案字宽动态钉定（见 offline_tab._pin_button_width），
    # 「文案恒 2 字」是防宽度跳动的另一道约束。

    # 圆角（Windows 11 风格）
    RADIUS_NORMAL = 4             # 标准圆角

    # 滑块控件高度（QSS 与 setFixedHeight 单源——改一处即全局生效）
    SLIDER_HEIGHT = 18

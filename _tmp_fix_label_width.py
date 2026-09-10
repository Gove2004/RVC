# 给实验功能Tab的所有标签设置固定宽度70px，确保滑动条左对齐
path = r"E:\Projects\Python\RVC\gui\infer\view\tabs\experimental_tab.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 添加 _lbl 辅助函数
old_import = """from gui.infer.view.widgets import _slrow, _sl_value_as_float, RangeSlider
from rvc.core.experimental import experimental_config"""

new_import = """from gui.infer.view.widgets import _slrow, _sl_value_as_float, RangeSlider
from rvc.core.experimental import experimental_config


def _lbl(text, width=70):
    \"\"\"创建固定宽度的标签，确保滑动条左对齐\"\"\"
    l = QLabel(text)
    l.setMinimumWidth(width)
    return l"""

content = content.replace(old_import, new_import)

# 替换所有 QLabel 为 _lbl
content = content.replace('g1.addWidget(QLabel("RMVPE 阈值"), r, 0)', 'g1.addWidget(_lbl("RMVPE 阈值"), r, 0)')
content = content.replace('g1.addWidget(QLabel("FCPE 阈值"), r, 0)', 'g1.addWidget(_lbl("FCPE 阈值"), r, 0)')
content = content.replace('g1.addWidget(QLabel("过渡区域"), r, 0)', 'g1.addWidget(_lbl("过渡区域"), r, 0)')
content = content.replace('g1.addWidget(QLabel("保护强度"), r, 0)', 'g1.addWidget(_lbl("保护强度"), r, 0)')
content = content.replace('gpm.addWidget(QLabel("原声音域"), r, 0)', 'gpm.addWidget(_lbl("原声音域"), r, 0)')
content = content.replace('gpm.addWidget(QLabel("目标音域"), r, 0)', 'gpm.addWidget(_lbl("目标音域"), r, 0)')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("experimental_tab.py: all labels fixed width 70px, sliders left-aligned")

"""音频 IO — 文件读写与设备枚举。"""
import os

# pythonw（GUI）下启动子进程会闪控制台窗口，仅 Windows 需要抑制
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW

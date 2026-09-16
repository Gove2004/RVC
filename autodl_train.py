"""RVC 云 GPU 训练向导 — 薄包装入口。

实际实现位于 autodl/ 包，本文件仅为向后兼容的启动入口。
等价于: python -m autodl
"""
import sys

from autodl.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

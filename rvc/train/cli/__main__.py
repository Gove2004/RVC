"""`python -m rvc.train.cli` 入口（等价于 `python app.py --train`）。"""
import sys

from rvc.train.cli.run import main

if __name__ == "__main__":
    sys.exit(main())

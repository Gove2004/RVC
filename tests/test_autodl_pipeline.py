"""autodl 向导管道金标准（S9）——逐行输出 + 退出码 diff。

覆盖路径：python -m autodl <数据集> → 环境体检 → 参数向导 → 配置确认 →
"开始训练？"喂 n 取消退出（rc=0）。管道喂答案，答案对齐本身也是行为基线：
问题数量/顺序变化会挪动答案落点，导致 diff 失败。

归一化（仅抹掉机器/时间相关噪声，语义字段全保留）：
- 时间戳 [MM-DD HH:MM:SS] → [TS]
- 体积/显存等数字+单位（123.4GB / 24.0 GB）→ <SIZE>
- 总耗时 00h00m02s → <ELAPSED>
- RVC_OUT_ROOT 与数据集目录的绝对路径 → <OUT_ROOT> / <DATASET>

采集:  python tests/test_autodl_pipeline.py capture
校验:  python tests/test_autodl_pipeline.py verify （unittest 亦自动发现）
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = PROJECT_ROOT / "tests" / "golden" / "autodl_pipeline.txt"

# 10 个空行（数据集路径取 argv、其余全取默认）+ n（取消训练）+ 冗余 n 防错位悬挂
STDIN_ANSWERS = "\n" * 10 + "n\n" + "n\n" * 20


def make_dataset_wav(path: Path, sr: int = 48000, dur: float = 2.0):
    """确定性正弦 wav（PCM16），文件大小/时长固定，素材统计输出可复现。"""
    import math
    import struct

    n = int(sr * dur)
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220.0 * i / sr)))
        for i in range(n)
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def normalize(text: str, out_root: Path, dataset_dir: Path) -> str:
    text = text.replace(str(out_root), "<OUT_ROOT>")
    text = text.replace(str(dataset_dir), "<DATASET>")
    text = re.sub(r"\[\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", "[TS]", text)
    text = re.sub(r"\d+(?:\.\d+)? ?(?:B|KB|MB|GB|TB)", "<SIZE>", text)
    text = re.sub(r"\d{2}h\d{2}m\d{2}s", "<ELAPSED>", text)
    return text


def run_wizard_pipe(out_root: Path, dataset_dir: Path) -> tuple[str, int]:
    """跑一次管道向导，返回 (归一化输出, 退出码)。

    cwd 用 out_root（≠ 项目根）——顺带验证 chdir 已从 import 时移入 main()
    后，向导在任意工作目录下仍能正确解析路径。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RVC_OUT_ROOT": str(out_root), "PYTHONPATH": str(PROJECT_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "autodl", str(dataset_dir)],
        input=STDIN_ANSWERS, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300,
        cwd=str(out_root), env=env,
    )
    return normalize(proc.stdout, out_root, dataset_dir), proc.returncode


class TestAutodlPipelineGolden(unittest.TestCase):
    def test_pipeline_output_and_exit_code_match_golden(self):
        self.assertTrue(GOLDEN_FILE.exists(), f"金标准缺失: {GOLDEN_FILE}（先跑 capture）")
        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td) / "out"
            dataset_dir = Path(td) / "dataset"
            dataset_dir.mkdir()
            make_dataset_wav(dataset_dir / "voice.wav")
            normalized, rc = run_wizard_pipe(out_root, dataset_dir)
        self.assertEqual(rc, 0, f"退出码不为 0（期望取消路径 rc=0）:\n{normalized}")
        golden = GOLDEN_FILE.read_text(encoding="utf-8")
        self.assertEqual(normalized, golden, "向导逐行输出与金标准不一致")


def capture():
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "out"
        dataset_dir = Path(td) / "dataset"
        dataset_dir.mkdir()
        make_dataset_wav(dataset_dir / "voice.wav")
        normalized, rc = run_wizard_pipe(out_root, dataset_dir)
    assert rc == 0, f"采集时退出码非 0:\n{normalized}"
    GOLDEN_FILE.write_text(normalized, encoding="utf-8")
    print(f"金标准已写入 {GOLDEN_FILE}（{len(normalized.splitlines())} 行）")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture()
    else:
        unittest.main()

"""运行时路径深度测试 — paths.py 各种路径常量、parse_sr、train_config_path。

覆盖：
- 路径常量：PROJECT_ROOT / ASSETS_ROOT / CONFIG_ROOT / STATE_FILE
- HUBERT_ROOT / RMVPE_PATH / PRETRAINED_ROOT / MODELS_DIR / TRAIN_LOGS_ROOT
- FFMPEG_EXE / FFPROBE_EXE / RESOURCES_DIR / ICON_PATH
- TRAIN_CONFIG_FILES / DEFAULT_TRAIN_SR
- parse_sr: 各种输入格式（'40k'/'48000'/40000/48.0/'48K'）
- train_config_path: 默认/指定采样率/不存在的采样率
- config_path
"""
import os
import unittest
from pathlib import Path

from rvc.runtime.paths import (
    ASSETS_ROOT,
    CONFIG_ROOT,
    DEFAULT_TRAIN_SR,
    FFMPEG_EXE,
    FFPROBE_EXE,
    HUBERT_ROOT,
    ICON_ACTIVE_PATH,
    ICON_IDLE_PATH,
    MODELS_DIR,
    PRETRAINED_ROOT,
    PROJECT_ROOT,
    RESOURCES_DIR,
    RMVPE_PATH,
    STATE_FILE,
    TRAIN_CONFIG_FILES,
    TRAIN_LOGS_ROOT,
    config_path,
    parse_sr,
    train_config_path,
)


class TestPathConstants(unittest.TestCase):
    """路径常量测试。"""

    def test_project_root_is_absolute(self):
        self.assertTrue(PROJECT_ROOT.is_absolute())

    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(PROJECT_ROOT.is_dir())

    def test_project_root_contains_rvc(self):
        self.assertTrue((PROJECT_ROOT / "rvc").exists())

    def test_project_root_contains_assets(self):
        self.assertTrue((PROJECT_ROOT / "assets").exists())

    def test_assets_root_is_subdir_of_project(self):
        self.assertEqual(ASSETS_ROOT, PROJECT_ROOT / "assets")

    def test_assets_root_exists(self):
        self.assertTrue(ASSETS_ROOT.exists())
        self.assertTrue(ASSETS_ROOT.is_dir())

    def test_config_root_is_subdir_of_assets(self):
        self.assertEqual(CONFIG_ROOT, ASSETS_ROOT / "configs")

    def test_state_file_is_in_config_root(self):
        self.assertEqual(STATE_FILE, CONFIG_ROOT / "save_state.json")

    def test_state_file_is_json(self):
        self.assertEqual(STATE_FILE.suffix, ".json")

    def test_hubert_root_is_subdir_of_assets(self):
        self.assertEqual(HUBERT_ROOT, ASSETS_ROOT / "hubert")

    def test_rmvpe_path_is_in_assets(self):
        self.assertEqual(RMVPE_PATH, ASSETS_ROOT / "rmvpe" / "rmvpe.pt")

    def test_rmvpe_path_suffix(self):
        self.assertEqual(RMVPE_PATH.suffix, ".pt")

    def test_pretrained_root_is_subdir_of_assets(self):
        self.assertEqual(PRETRAINED_ROOT, ASSETS_ROOT / "pretrained")

    def test_models_dir_is_subdir_of_assets(self):
        self.assertEqual(MODELS_DIR, ASSETS_ROOT / "models")

    def test_train_logs_root_is_subdir_of_project(self):
        self.assertEqual(TRAIN_LOGS_ROOT, PROJECT_ROOT / "logs")

    def test_ffmpeg_exe_path(self):
        self.assertEqual(FFMPEG_EXE, ASSETS_ROOT / "ffmpeg" / "ffmpeg.exe")

    def test_ffmpeg_exe_suffix(self):
        self.assertEqual(FFMPEG_EXE.suffix, ".exe")

    def test_ffprobe_exe_path(self):
        self.assertEqual(FFPROBE_EXE, ASSETS_ROOT / "ffmpeg" / "ffprobe.exe")

    def test_resources_dir_is_subdir_of_assets(self):
        self.assertEqual(RESOURCES_DIR, ASSETS_ROOT / "resources")

    def test_icon_idle_path(self):
        self.assertEqual(ICON_IDLE_PATH, RESOURCES_DIR / "icon_idle.png")

    def test_icon_active_path(self):
        self.assertEqual(ICON_ACTIVE_PATH, RESOURCES_DIR / "icon_active.png")

    def test_icon_paths_are_png(self):
        self.assertEqual(ICON_IDLE_PATH.suffix, ".png")
        self.assertEqual(ICON_ACTIVE_PATH.suffix, ".png")


class TestTrainConfigFiles(unittest.TestCase):
    """训练配置文件映射测试。"""

    def test_contains_40k(self):
        self.assertIn(40000, TRAIN_CONFIG_FILES)

    def test_contains_48k(self):
        self.assertIn(48000, TRAIN_CONFIG_FILES)

    def test_40k_path(self):
        path = TRAIN_CONFIG_FILES[40000]
        self.assertEqual(path, CONFIG_ROOT / "40ktrain_config.json")

    def test_48k_path(self):
        path = TRAIN_CONFIG_FILES[48000]
        self.assertEqual(path, CONFIG_ROOT / "48ktrain_config.json")

    def test_paths_are_json(self):
        for path in TRAIN_CONFIG_FILES.values():
            self.assertEqual(path.suffix, ".json")

    def test_default_train_sr_is_48000(self):
        self.assertEqual(DEFAULT_TRAIN_SR, 48000)

    def test_default_train_sr_in_config_files(self):
        self.assertIn(DEFAULT_TRAIN_SR, TRAIN_CONFIG_FILES)


class TestParseSr(unittest.TestCase):
    """parse_sr 函数测试。"""

    def test_40k_string(self):
        self.assertEqual(parse_sr("40k"), 40000)

    def test_48k_string(self):
        self.assertEqual(parse_sr("48k"), 48000)

    def test_32k_string(self):
        self.assertEqual(parse_sr("32k"), 32000)

    def test_16k_string(self):
        self.assertEqual(parse_sr("16k"), 16000)

    def test_8k_string(self):
        self.assertEqual(parse_sr("8k"), 8000)

    def test_uppercase_k(self):
        self.assertEqual(parse_sr("48K"), 48000)

    def test_mixed_case(self):
        self.assertEqual(parse_sr("40K"), 40000)

    def test_decimal_k(self):
        self.assertEqual(parse_sr("44.1k"), 44100)

    def test_decimal_k_2(self):
        self.assertEqual(parse_sr("22.05k"), 22050)

    def test_integer_string(self):
        self.assertEqual(parse_sr("48000"), 48000)

    def test_integer_string_2(self):
        self.assertEqual(parse_sr("40000"), 40000)

    def test_integer_input(self):
        self.assertEqual(parse_sr(48000), 48000)

    def test_integer_input_2(self):
        self.assertEqual(parse_sr(40000), 40000)

    def test_float_input(self):
        self.assertEqual(parse_sr(48000.0), 48000)

    def test_float_input_2(self):
        self.assertEqual(parse_sr(44100.0), 44100)

    def test_string_with_spaces(self):
        self.assertEqual(parse_sr("  48k  "), 48000)

    def test_string_with_leading_spaces(self):
        self.assertEqual(parse_sr("   40000"), 40000)

    def test_string_with_trailing_spaces(self):
        self.assertEqual(parse_sr("48000   "), 48000)

    def test_96k(self):
        self.assertEqual(parse_sr("96k"), 96000)

    def test_192k(self):
        self.assertEqual(parse_sr("192k"), 192000)

    def test_large_integer(self):
        self.assertEqual(parse_sr(192000), 192000)

    def test_zero(self):
        self.assertEqual(parse_sr(0), 0)

    def test_zero_string(self):
        self.assertEqual(parse_sr("0"), 0)

    def test_zero_k(self):
        self.assertEqual(parse_sr("0k"), 0)

    def test_negative(self):
        """负数也能解析（虽然不合法，但函数不做校验）。"""
        self.assertEqual(parse_sr(-48000), -48000)

    def test_various_common_sample_rates(self):
        """常见采样率全部测试。"""
        expected = {
            "8k": 8000,
            "16k": 16000,
            "22.05k": 22050,
            "32k": 32000,
            "44.1k": 44100,
            "48k": 48000,
            "96k": 96000,
            "192k": 192000,
        }
        for sr_str, sr_int in expected.items():
            self.assertEqual(parse_sr(sr_str), sr_int, f"parse_sr('{sr_str}') should be {sr_int}")


class TestConfigPath(unittest.TestCase):
    """config_path 函数测试。"""

    def test_returns_state_file(self):
        self.assertEqual(config_path(), STATE_FILE)

    def test_returns_path_object(self):
        self.assertIsInstance(config_path(), Path)

    def test_returns_absolute_path(self):
        self.assertTrue(config_path().is_absolute())


class TestTrainConfigPath(unittest.TestCase):
    """train_config_path 函数测试。"""

    def test_default_returns_48k(self):
        """sr=None 时返回默认 48k 配置。"""
        path = train_config_path()
        self.assertEqual(path, TRAIN_CONFIG_FILES[48000])

    def test_48k_returns_48k(self):
        path = train_config_path(48000)
        self.assertEqual(path, TRAIN_CONFIG_FILES[48000])

    def test_40k_returns_40k(self):
        path = train_config_path(40000)
        self.assertEqual(path, TRAIN_CONFIG_FILES[40000])

    def test_string_48k(self):
        path = train_config_path("48k")
        self.assertEqual(path, TRAIN_CONFIG_FILES[48000])

    def test_string_40k(self):
        path = train_config_path("40k")
        self.assertEqual(path, TRAIN_CONFIG_FILES[40000])

    def test_unsupported_sr_falls_back_to_default(self):
        """不支持的采样率回退到默认 48k。"""
        path = train_config_path(96000)
        self.assertEqual(path, TRAIN_CONFIG_FILES[48000])

    def test_unsupported_string_falls_back(self):
        path = train_config_path("96k")
        self.assertEqual(path, TRAIN_CONFIG_FILES[48000])

    def test_returns_path_object(self):
        self.assertIsInstance(train_config_path(), Path)

    def test_returns_absolute_path(self):
        self.assertTrue(train_config_path().is_absolute())

    def test_default_config_exists(self):
        """默认 48k 配置文件应该存在。"""
        path = train_config_path()
        self.assertTrue(path.exists())

    def test_config_file_is_json(self):
        path = train_config_path()
        self.assertEqual(path.suffix, ".json")


class TestPathConsistency(unittest.TestCase):
    """路径一致性测试。"""

    def test_all_asset_paths_under_assets_root(self):
        """所有 assets 下的路径都在 ASSETS_ROOT 下。"""
        asset_paths = [
            CONFIG_ROOT, HUBERT_ROOT, RMVPE_PATH.parent,
            PRETRAINED_ROOT, MODELS_DIR, FFMPEG_EXE.parent,
            RESOURCES_DIR,
        ]
        for path in asset_paths:
            self.assertTrue(
                str(path).startswith(str(ASSETS_ROOT)),
                f"{path} should be under {ASSETS_ROOT}"
            )

    def test_no_hardcoded_absolute_paths_in_constants(self):
        """路径常量都是基于 PROJECT_ROOT 计算的，不是硬编码的绝对路径。"""
        # 检查路径不包含用户特定的路径前缀（简单检查）
        self.assertTrue(str(PROJECT_ROOT).endswith("RVC"))

    def test_train_logs_not_under_assets(self):
        """训练日志在项目根下，不在 assets 下。"""
        self.assertFalse(str(TRAIN_LOGS_ROOT).startswith(str(ASSETS_ROOT)))
        self.assertTrue(str(TRAIN_LOGS_ROOT).startswith(str(PROJECT_ROOT)))

    def test_ffmpeg_windows_specific(self):
        """ffmpeg 路径是 .exe（Windows 专用）。"""
        self.assertEqual(FFMPEG_EXE.suffix, ".exe")
        self.assertEqual(FFPROBE_EXE.suffix, ".exe")


if __name__ == "__main__":
    unittest.main()

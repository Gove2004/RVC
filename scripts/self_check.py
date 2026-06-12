"""Self-check script to validate refactored architecture."""
from pathlib import Path

from configs.config import train_config_path, state_path, load_state_json, save_state_json
from rvc.params import Params
from rvc.inference_cache import InferenceCache
from gui.infer.controller import InferController, ModelConfig


def check_config_paths():
    """Verify config path resolution works."""
    print("[1/5] Checking config path resolution...")

    path_32k = train_config_path("32k")
    path_48k = train_config_path(48000)
    assert path_32k.exists(), f"32k config not found: {path_32k}"
    assert path_48k.exists(), f"48k config not found: {path_48k}"

    state_gui = state_path("gui")
    state_models = state_path("models")
    state_train = state_path("train")
    print(f"  [OK] Train configs: {path_32k.parent}")
    print(f"  [OK] State files: {state_gui.parent}")


def check_state_json():
    """Verify state JSON load/save works."""
    print("[2/5] Checking state JSON operations...")

    test_data = {"test": "value", "number": 42}
    save_state_json("gui", test_data)
    loaded = load_state_json("gui", {})
    assert loaded.get("test") == "value", f"State save/load failed: {loaded}"
    print(f"  [OK] State persistence working")


def check_params_update():
    """Verify Params can be updated."""
    print("[3/5] Checking runtime params...")

    params = Params()
    original_pitch = params.pitch
    params.pitch = 12
    assert params.pitch == 12, "Params update failed"
    params.pitch = original_pitch
    print(f"  [OK] Runtime params mutable")


def check_controller():
    """Verify controller pattern works."""
    print("[4/5] Checking controller instantiation...")

    cache = InferenceCache()
    params = Params()
    controller = InferController(runtime_params=params, inference_cache=cache)

    model_cfg = ModelConfig(
        pitch=12,
        index_rate=0.0,
        rms_mix=0.5,
        gender=0.0,
        protect=0.33,
        f0method="fcpe",
    )
    controller.apply_model_config(model_cfg)
    assert controller.runtime_params.pitch == 12, "Controller config apply failed"
    print(f"  [OK] Controller pattern working")


def check_cache():
    """Verify inference cache operations."""
    print("[5/5] Checking inference cache...")

    cache = InferenceCache()
    assert cache.get_hubert("test") is None, "Empty cache should return None"

    dummy_model = object()
    cache.set_hubert("test", dummy_model)
    assert cache.get_hubert("test") is dummy_model, "Cache set/get failed"
    print(f"  [OK] Inference cache working")


if __name__ == "__main__":
    print("Running self-check...\n")
    check_config_paths()
    check_state_json()
    check_params_update()
    check_controller()
    check_cache()
    print("\n[SUCCESS] All checks passed!")

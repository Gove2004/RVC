"""Checkpoint helpers shared by inference and training frontends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


STATE_DICT_KEYS = ("state", "state_dict", "model_state_dict")


def unwrap_state_dict(checkpoint: Any) -> Any:
    """Return the model state dict from common MSS checkpoint containers."""
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    model_type: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
) -> Any:
    """Load a checkpoint package."""
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def load_state_dict(
    path: str | Path,
    *,
    model_type: str | None = None,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
) -> Any:
    """Load and unwrap the model state dict from a checkpoint file."""
    return unwrap_state_dict(
        load_checkpoint(
            path,
            model_type=model_type,
            map_location=map_location,
            weights_only=weights_only,
        )
    )


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_or_path: Any,
    *,
    model_type: str | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load weights from a checkpoint package or file into a model."""
    if isinstance(checkpoint_or_path, (str, Path)):
        state_dict = load_state_dict(checkpoint_or_path, model_type=model_type, map_location=map_location)
    else:
        state_dict = unwrap_state_dict(checkpoint_or_path)
    return model.load_state_dict(state_dict, strict=strict)

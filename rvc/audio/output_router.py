"""实时背景音混合与输出路由。"""
import logging
import queue

import numpy as np
import torch

logger = logging.getLogger(__name__)


def mix_bgm(
    chunk: torch.Tensor,
    bgm_audio: torch.Tensor | None,
    bgm_ptr: int,
    bgm_mix_buffer: torch.Tensor,
    bgm_volume: float,
    block_frame: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    if bgm_audio is None or bgm_volume <= 0:
        return chunk, bgm_audio, bgm_ptr

    if bgm_audio.device != bgm_mix_buffer.device:
        bgm_audio = bgm_audio.to(bgm_mix_buffer.device)

    audio_length = bgm_audio.shape[0]
    need = block_frame
    cursor = 0
    bgm_mix_buffer.zero_()
    while need > 0:
        take = min(need, audio_length - bgm_ptr)
        bgm_mix_buffer[cursor:cursor + take] = bgm_audio[bgm_ptr:bgm_ptr + take]
        bgm_ptr = (bgm_ptr + take) % audio_length
        cursor += take
        need -= take

    return chunk + bgm_mix_buffer * bgm_volume, bgm_audio, bgm_ptr


def write_main_output(chunk: torch.Tensor, outdata: np.ndarray, channels: int) -> None:
    out_chunk = chunk.cpu().numpy()
    if channels == 1:
        outdata[:, 0] = out_chunk
    else:
        outdata[:] = out_chunk[:, None]


def route_secondary_output(outdata: np.ndarray, stream2, out2_q: queue.Queue, enable_out2: bool) -> bool:
    if stream2 and enable_out2:
        if out2_q.full():
            try:
                out2_q.get_nowait()
            except Exception:
                pass
        out2_q.put_nowait(outdata.copy())
        return False

    if stream2 and not enable_out2:
        return True
    return False

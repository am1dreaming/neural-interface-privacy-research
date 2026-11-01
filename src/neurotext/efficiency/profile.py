"""Cost of a model on a wearable budget: parameters, bytes, latency; and input reductions
(fewer channels, lower sampling rate) used for ablations.

Latency measured on a laptop/server CPU is only a *relative* proxy. Real wearable numbers
need the target hardware.
"""

from __future__ import annotations

import io
import time

import numpy as np


def count_parameters(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def serialized_size_bytes(model) -> int:
    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def quantized_size_bytes(model) -> int | None:
    """Measured size after int8 dynamic quantization of Linear layers (convs stay float).

    Returns None if this PyTorch build has no working quantization engine (e.g. the
    default macOS wheels raise `NoQEngine`). Use `estimate_int8_linear_bytes` then.
    """
    import copy
    import warnings

    import torch

    supported = list(torch.backends.quantized.supported_engines)
    previous = torch.backends.quantized.engine
    try:
        for engine in ("fbgemm", "x86", "qnnpack", "onednn"):
            if engine not in supported:
                continue
            try:
                torch.backends.quantized.engine = engine
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    q = torch.ao.quantization.quantize_dynamic(
                        copy.deepcopy(model).eval(), {torch.nn.Linear}, dtype=torch.qint8
                    )
                return serialized_size_bytes(q)
            except RuntimeError:
                continue
        return None
    finally:
        try:
            torch.backends.quantized.engine = previous
        except RuntimeError:
            pass


def estimate_int8_linear_bytes(model) -> int:
    """Analytic estimate of the same quantity, available on any platform.

    Plain `nn.Linear` weights count 1 byte each plus a 4-byte scale per layer; everything
    else (biases, convs, norms, attention projections, positional parameters) stays fp32.
    This is an estimate, not a measurement: file-format overhead is ignored.
    """
    import torch

    int8_params = set()
    total = 0
    for m in model.modules():
        if type(m) is torch.nn.Linear:  # exact type: subclasses are not swapped by quantize_dynamic
            int8_params.add(id(m.weight))
            total += m.weight.numel() + 4
    for p in model.parameters():
        if id(p) not in int8_params:
            total += p.numel() * 4
    return total


def measure_latency_ms(model, example: np.ndarray, repeats: int = 20, warmup: int = 3) -> float:
    """Median forward-pass time in ms for one batch."""
    import torch

    x = torch.as_tensor(example, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def select_channels(x: np.ndarray, channel_idx) -> np.ndarray:
    """Keep a subset of channels, (N, C, T) -> (N, len(channel_idx), T)."""
    return x[:, list(channel_idx), :]


def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Anti-aliased downsampling along time by an integer factor."""
    from scipy.signal import decimate

    if factor == 1:
        return x
    return decimate(x, factor, axis=-1, zero_phase=True).astype(x.dtype)

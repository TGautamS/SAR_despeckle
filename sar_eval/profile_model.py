"""
profile_model.py
================
Efficiency block: parameters, FLOPs, inference time (ms / 256x256 image).
This is the column where your Mamba model WINS -- against SAR-DDPM especially.

torch is imported lazily here so the rest of the harness needs no torch at all.

Usage (in your own script, after you've built/loaded each model):

    from profile_model import profile_model
    row = profile_model(model, input_shape=(1, 1, 256, 256), device="cuda")
    print(row)   # {'params_M':..., 'gflops':..., 'time_ms_mean':..., ...}

For SAR-DDPM: measure ONE timing run and report it as an efficiency-only row.
The expected story: "SAR-DDPM = seconds-minutes/scene (iterative sampling);
ours = milliseconds at comparable quality."
"""

from __future__ import annotations


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6  # millions


def profile_model(model, input_shape=(1, 1, 256, 256), device="cuda",
                  warmup: int = 10, runs: int = 50) -> dict:
    import torch, time
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = model.to(dev).eval()
    x = torch.randn(*input_shape, device=dev)

    out = {"params_M": round(count_params(model), 4), "device": dev,
           "input_shape": str(tuple(input_shape))}

    # FLOPs (optional)
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(x,), verbose=False)
        out["gflops"] = round(flops / 1e9, 4)
    except Exception as e:
        out["gflops"] = None
        out["gflops_note"] = f"thop unavailable ({type(e).__name__}); pip install thop"

    # timing
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / runs

    out["time_ms_mean"] = round(elapsed * 1000.0, 4)
    out["fps"] = round(1.0 / elapsed, 2)
    return out

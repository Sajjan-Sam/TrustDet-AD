"""Shared GPU timing utilities used by all run_*.py detector scripts."""
import time
import torch


def get_gpu_name(device):
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return "CPU"


def reset_gpu_memory_stats(device):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_mb(device):
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    return 0.0


class LatencyTimer:
    """Times single-image forward passes. CUDA-synchronized for accurate GPU timing."""

    def __init__(self, device):
        self.device = device
        self.samples_ms = []

    def time_call(self, fn, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.samples_ms.append(elapsed_ms)
        return result

    def mean_latency_ms(self):
        if not self.samples_ms:
            return 0.0
        return round(sum(self.samples_ms) / len(self.samples_ms), 4)

    def fps(self):
        mean_ms = self.mean_latency_ms()
        if mean_ms <= 0:
            return 0.0
        return round(1000.0 / mean_ms, 4)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config_loader import load_paths

    paths = load_paths()
    device = f"cuda:{paths['gpu_id']}"
    print(f"GPU: {get_gpu_name(device)}")
    print(f"Peak memory (MB): {get_peak_gpu_memory_mb(device)}")

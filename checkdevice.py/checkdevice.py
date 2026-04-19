"""
Hardware sanity check for PyTorch on Apple Silicon.
Run: uv run python check_device.py
"""

import platform
import time

import torch


def get_device() -> torch.device:
    """Return the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark(device: torch.device, size: int = 4096, iters: int = 10) -> float:
    """Matrix multiplication benchmark. Returns seconds per iter."""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(3):
        _ = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


def main() -> None:
    print("=" * 60)
    print("System")
    print("=" * 60)
    print(f"Platform:        {platform.platform()}")
    print(f"Processor:       {platform.processor()}")
    print(f"Python:          {platform.python_version()}")
    print(f"PyTorch:         {torch.__version__}")
    print()

    print("=" * 60)
    print("Backends")
    print("=" * 60)
    print(f"MPS available:   {torch.backends.mps.is_available()}")
    print(f"MPS built:       {torch.backends.mps.is_built()}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    print()

    device = get_device()
    print(f"Selected device: {device}")
    print()

    print("=" * 60)
    print("Benchmark: 4096x4096 matmul")
    print("=" * 60)

    cpu_time = benchmark(torch.device("cpu"), size=2048, iters=3)
    print(f"CPU (2048x2048): {cpu_time * 1000:.2f} ms/iter")

    if device.type != "cpu":
        acc_time = benchmark(device, size=4096, iters=10)
        print(f"{device.type.upper()} (4096x4096): {acc_time * 1000:.2f} ms/iter")

    print()
    print("All good. You're ready to train.")


if __name__ == "__main__":
    main()

"""Device detection utility for Mac M3 Pro (MPS) and CUDA."""

import torch


def get_device() -> torch.device:
    """Get the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration)")
    return device


if __name__ == "__main__":
    device = get_device()
    
    # Quick benchmark
    x = torch.randn(100, 256, 50, 50, device=device)
    y = torch.randn(100, 256, 50, 50, device=device)
    
    import time
    start = time.time()
    for _ in range(10):
        z = x @ y.transpose(-2, -1)
    if device.type != "cpu":
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Benchmark: {elapsed:.3f}s for 10 matmuls on {device}")

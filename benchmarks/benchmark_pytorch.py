import argparse
import io
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def get_memory_stats_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) from /proc/self/status and resource."""
    rss_kb = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = int(line.split()[1])
            break
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss_kb / 1024.0, 2), round(peak_kb / 1024.0, 2)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Measure the serialized PyTorch model weights size in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return round(buffer.tell() / (1024.0 * 1024.0), 2)


def benchmark_pytorch(
    warmup_runs: int = 20,
    num_runs: int = 100,
    num_threads: int = 1,
    inter_threads: int = 1,
) -> dict:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(inter_threads)
    baseline_rss_mb, _ = get_memory_stats_mb()
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    model_size_mb = get_model_size_mb(model)

    image_path = (
        Path(__file__).resolve().parent.parent / "data" / "sample_images" / "cat.jpg"
    )
    img = Image.open(image_path).convert("RGB")
    input_batch = weights.transforms()(img).unsqueeze(0)
    # Warm-up iterations
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_batch)

    # Timed inference runs
    latencies_ms = []
    start_total = time.perf_counter()
    start_cpu = time.process_time()
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            output = model(input_batch)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_time_sec = time.perf_counter() - start_total
    cpu_time_sec = time.process_time() - start_cpu

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.max(probs, 0)
    batch_size = int(input_batch.shape[0])
    lat_array = np.array(latencies_ms)
    current_rss_mb, peak_rss_mb = get_memory_stats_mb()
    return {
        "framework": "PyTorch (CPU)",
        "config": f"intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}",
        "batch_size": batch_size,
        "input_shape": f"{tuple(input_batch.shape)}",
        "avg_latency_ms": round(float(np.mean(lat_array)), 3),
        "p50_latency_ms": round(float(np.percentile(lat_array, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(lat_array, 95)), 3),
        "throughput_img_per_sec": round(
            float((num_runs * batch_size) / total_time_sec), 2
        ),
        "cpu_util_pct": round((cpu_time_sec / total_time_sec) * 100.0, 1),
        "model_size_mb": model_size_mb,
        "baseline_rss_mb": baseline_rss_mb,
        "model_rss_delta_mb": round(max(0.0, current_rss_mb - baseline_rss_mb), 2),
        "peak_memory_mb": peak_rss_mb,
        "top1_class": weights.meta["categories"][top_catid.item()],
        "top1_conf_pct": round(top_prob.item() * 100.0, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch CPU inference.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = benchmark_pytorch(args.warmup, args.runs, args.threads, args.inter_threads)
    if args.json:
        print(json.dumps(result))
    else:
        for k, v in result.items():
            print(f"{k:<24}: {v}")

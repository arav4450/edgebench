import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def get_memory_stats_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) from /proc/self/status and resource."""
    rss_kb = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = int(line.split()[1])
            break
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss_kb / 1024.0, 2), round(peak_kb / 1024.0, 2)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Resize shorter edge to 256, center-crop 224x224, and apply ImageNet normalization."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w, new_h = (256, int(256 * h / w)) if w <= h else (int(256 * w / h), 256)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = int(round((new_w - 224) / 2.0))
    top = int(round((new_h - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def benchmark_onnx(
    warmup_runs: int = 20,
    num_runs: int = 100,
    intra_threads: int = 1,
    inter_threads: int = 1,
    mode: str = "SEQUENTIAL",
) -> dict:
    baseline_rss_mb, _ = get_memory_stats_mb()
    project_root = Path(__file__).resolve().parent.parent
    onnx_path = project_root / "models" / "mobilenet_v2.onnx"
    model_size_mb = round(onnx_path.stat().st_size / (1024.0 * 1024.0), 2)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = inter_threads
    sess_options.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL
        if mode == "PARALLEL"
        else ort.ExecutionMode.ORT_SEQUENTIAL
    )

    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    input_batch = preprocess_image(project_root / "data" / "sample_images" / "cat.jpg")
    # Warm-up iterations
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_batch})

    # Timed inference runs
    latencies_ms = []
    start_total = time.perf_counter()
    start_cpu = time.process_time()
    for _ in range(num_runs):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: input_batch})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_time_sec = time.perf_counter() - start_total
    cpu_time_sec = time.process_time() - start_cpu

    logits = outputs[0][0]
    exp_scores = np.exp(logits - np.max(logits))
    probs = exp_scores / np.sum(exp_scores)
    top_catid = int(np.argmax(probs))
    batch_size = int(input_batch.shape[0])

    lat_array = np.array(latencies_ms)
    current_rss_mb, peak_rss_mb = get_memory_stats_mb()
    intra_label = "auto" if intra_threads == 0 else str(intra_threads)
    return {
        "framework": "ONNX Runtime (CPU)",
        "config": f"intra={intra_label}, inter={inter_threads}, mode={mode}",
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
        "top1_class": "Egyptian cat" if top_catid == 285 else f"class_{top_catid}",
        "top1_conf_pct": round(float(probs[top_catid]) * 100.0, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime CPU inference."
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--intra-threads", type=int, default=1)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="SEQUENTIAL", choices=["SEQUENTIAL", "PARALLEL"]
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = benchmark_onnx(
        args.warmup, args.runs, args.intra_threads, args.inter_threads, args.mode
    )
    if args.json:
        print(json.dumps(result))
    else:
        for k, v in result.items():
            print(f"{k:<24}: {v}")

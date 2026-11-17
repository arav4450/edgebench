import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_onnx import get_memory_stats_mb, preprocess_image

# Map the 10 Imagenette WordNet folder IDs to standard 1000-class ImageNet indices
WNID_TO_IMAGENET_IDX = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def profile_single_model(
    model_path: Path, threads: int = 4, warmup: int = 20, runs: int = 100
) -> dict:
    """Measure model size, latency (avg, p50, p95), throughput, and memory in an isolated process."""
    baseline_rss, _ = get_memory_stats_mb()
    model_size_mb = round(model_path.stat().st_size / (1024.0 * 1024.0), 2)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    sample_img = (
        Path(__file__).resolve().parent.parent / "data" / "sample_images" / "cat.jpg"
    )
    batch = preprocess_image(sample_img)

    for _ in range(warmup):
        _ = session.run(None, {input_name: batch})

    latencies_ms = []
    t_start = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: batch})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_sec = time.perf_counter() - t_start

    rss_mb, peak_mb = get_memory_stats_mb()
    lat = np.array(latencies_ms)
    return {
        "model_size_mb": model_size_mb,
        "avg_latency_ms": round(float(np.mean(lat)), 3),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 3),
        "throughput_img_per_sec": round(float(runs / total_sec), 2),
        "rss_mb": rss_mb,
        "model_rss_delta_mb": round(max(0.0, rss_mb - baseline_rss), 2),
        "peak_memory_mb": peak_mb,
    }


def evaluate_imagenette(
    fp32_path: Path, int8_path: Path, data_root: Path, per_class: int = 50
) -> dict:
    """Evaluate FP32/INT8 Top-1 Accuracy and Prediction Agreement on Imagenette validation split."""
    from torchvision.datasets import Imagenette

    extracted = data_root / "imagenette2-160"
    val_ds = Imagenette(
        root=str(data_root), split="val", size="160px", download=not extracted.exists()
    )

    # Sample `per_class` validation images from each of the 10 classes (500 total)
    counts: dict[int, int] = {}
    samples: list[tuple[Path, int]] = []
    for img_path, ds_label in val_ds._samples:
        if counts.get(ds_label, 0) < per_class:
            wnid = val_ds.wnids[ds_label]
            samples.append((Path(img_path), WNID_TO_IMAGENET_IDX[wnid]))
            counts[ds_label] = counts.get(ds_label, 0) + 1

    fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    in_fp32 = fp32_sess.get_inputs()[0].name
    in_int8 = int8_sess.get_inputs()[0].name

    fp32_correct = 0
    int8_correct = 0
    agree_count = 0

    for img_path, true_idx in samples:
        batch = preprocess_image(img_path)
        fp32_pred = int(np.argmax(fp32_sess.run(None, {in_fp32: batch})[0][0]))
        int8_pred = int(np.argmax(int8_sess.run(None, {in_int8: batch})[0][0]))

        if fp32_pred == true_idx:
            fp32_correct += 1
        if int8_pred == true_idx:
            int8_correct += 1
        if fp32_pred == int8_pred:
            agree_count += 1

    n = len(samples)
    return {
        "eval_samples": n,
        "fp32_top1_acc_pct": round((fp32_correct / n) * 100.0, 2),
        "int8_top1_acc_pct": round((int8_correct / n) * 100.0, 2),
        "prediction_agreement_pct": round((agree_count / n) * 100.0, 2),
    }


def run_trials_isolated(
    script_path: Path, model_path: Path, precision: str, threads: int, trials: int = 5
) -> dict:
    """Run `profile_single_model` across `trials` isolated subprocesses."""
    runs = []
    for _ in range(trials):
        cmd = [
            sys.executable,
            str(script_path),
            "--single-model",
            str(model_path),
            "--threads",
            str(threads),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    agg = {
        "precision": precision,
        "threads": threads,
        "trials": trials,
        "model_size_mb": runs[0]["model_size_mb"],
        "avg_latency_ms": round(float(np.mean([r["avg_latency_ms"] for r in runs])), 3),
        "trial_std_ms": round(float(np.std([r["avg_latency_ms"] for r in runs])), 3),
    }
    for k in [
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_img_per_sec",
        "rss_mb",
        "model_rss_delta_mb",
        "peak_memory_mb",
    ]:
        agg[k] = round(float(np.mean([r[k] for r in runs])), 3 if "ms" in k else 2)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs INT8 MobileNetV2.")
    parser.add_argument("--single-model", type=str, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.single_model:
        print(
            json.dumps(
                profile_single_model(Path(args.single_model), threads=args.threads)
            )
        )
        return

    root = Path(__file__).resolve().parent.parent
    fp32_path = root / "models" / "mobilenet_v2.onnx"
    int8_path = root / "models" / "mobilenet_v2_int8.onnx"
    data_root = root / "data" / "imagenette"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not int8_path.is_file():
        subprocess.run(
            [sys.executable, str(root / "src" / "quantize_model.py")], check=True
        )

    print(
        "\nEvaluating Imagenette Top-1 Accuracy & FP32/INT8 Prediction Agreement (500 val images)..."
    )
    acc = evaluate_imagenette(fp32_path, int8_path, data_root, per_class=50)

    results = []
    for threads in [1, 4]:
        for prec, path in [("FP32", fp32_path), ("INT8", int8_path)]:
            print(f"Benchmarking {prec} model (threads={threads}, 5 trials)...")
            row = run_trials_isolated(Path(__file__), path, prec, threads, trials=5)
            row["top1_accuracy_pct"] = (
                acc["fp32_top1_acc_pct"] if prec == "FP32" else acc["int8_top1_acc_pct"]
            )
            row["prediction_agreement_pct"] = (
                100.0 if prec == "FP32" else acc["prediction_agreement_pct"]
            )
            results.append(row)

    header = (
        f"{'Precision':<9} | {'Thr':>3} | {'Size(MB)':>8} | {'Avg±Std(ms)':>13} | "
        f"{'p50(ms)':>7} | {'p95(ms)':>7} | {'Img/s':>7} | {'RSS/Peak(MB)':>13} | {'Top-1 Acc':>9} | {'Agree vs FP32':>13}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        avg_s = f"{r['avg_latency_ms']:>6.2f}±{r['trial_std_ms']:<5.2f}"
        mem_s = f"{r['rss_mb']:.1f}/{r['peak_memory_mb']:.1f}"
        print(
            f"{r['precision']:<9} | {r['threads']:>3} | {r['model_size_mb']:>8.2f} | {avg_s:>13} | "
            f"{r['p50_latency_ms']:>7.2f} | {r['p95_latency_ms']:>7.2f} | "
            f"{r['throughput_img_per_sec']:>7.1f} | {mem_s:>13} | "
            f"{r['top1_accuracy_pct']:>8.2f}% | {r['prediction_agreement_pct']:>12.2f}%"
        )
    print("-" * len(header))

    with (results_dir / "quantization_benchmark_results.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (results_dir / "quantization_benchmark_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(
        f"\nSaved quantization results to {results_dir / 'quantization_benchmark_results.csv'}"
    )


if __name__ == "__main__":
    main()

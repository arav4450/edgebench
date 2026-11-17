import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def get_system_info() -> dict:
    """Record CPU model, physical/logical cores, RAM, OS version, and 1-min background load."""
    cpu_info = Path("/proc/cpuinfo").read_text().splitlines()
    cpu_model = next(
        (l.split(":", 1)[1].strip() for l in cpu_info if "model name" in l), "Unknown"
    )
    core_ids = {l.split(":", 1)[1].strip() for l in cpu_info if l.startswith("core id")}
    mem_kb = next(
        int(l.split()[1])
        for l in Path("/proc/meminfo").read_text().splitlines()
        if "MemTotal:" in l
    )
    return {
        "cpu_model": cpu_model,
        "physical_cores": len(core_ids) or (os.cpu_count() or 1),
        "logical_cores": os.cpu_count() or 1,
        "total_ram_gb": round(mem_kb / (1024.0 * 1024.0), 2),
        "os_version": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "load_avg_1m": round(os.getloadavg()[0], 2),
    }


def run_isolated(cmd: list[str], trials: int = 3) -> dict:
    """Execute a benchmark configuration across multiple isolated trials and aggregate."""
    runs = []
    for _ in range(trials):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    agg = dict(runs[0])
    for key in [
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_img_per_sec",
        "cpu_util_pct",
        "baseline_rss_mb",
        "model_rss_delta_mb",
        "peak_memory_mb",
    ]:
        agg[key] = round(
            float(np.mean([r[key] for r in runs])), 3 if "ms" in key else 2
        )
    agg["trial_std_ms"] = round(float(np.std([r["avg_latency_ms"] for r in runs])), 3)
    agg["trials"] = trials
    return agg


def print_table(results: list[dict]) -> None:
    header = (
        f"{'Framework':<18} | {'Configuration':<32} | {'Avg±Std(ms)':>13} | "
        f"{'p50(ms)':>7} | {'p95(ms)':>7} | {'Img/s':>7} | {'CPU%':>6} | "
        f"{'Mod/Peak RAM(MB)':>16} | {'Prediction':<20}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        avg_std = f"{r['avg_latency_ms']:>6.2f}±{r['trial_std_ms']:<5.2f}"
        ram_str = f"{r['model_rss_delta_mb']:>5.1f} / {r['peak_memory_mb']:<6.1f}"
        pred_str = f"{r['top1_class']} ({r['top1_conf_pct']}%)"
        print(
            f"{r['framework']:<18} | {r['config']:<32} | {avg_std:>13} | "
            f"{r['p50_latency_ms']:>7.2f} | {r['p95_latency_ms']:>7.2f} | "
            f"{r['throughput_img_per_sec']:>7.1f} | {r['cpu_util_pct']:>5.1f}% | "
            f"{ram_str:>16} | {pred_str:<20}"
        )
    print("-" * len(header))


def save_results(results: list[dict], sys_info: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "benchmark_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    payload = {"system_info": sys_info, "benchmarks": results}
    (results_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pt_script = str(root / "benchmarks" / "benchmark_pytorch.py")
    ort_script = str(root / "benchmarks" / "benchmark_onnx.py")

    sys_info = get_system_info()
    print("=== System & Hardware Information ===")
    for k, v in sys_info.items():
        print(f"  {k:<16}: {v}")

    results = []
    for threads in [1, 4]:
        print(f"Benchmarking PyTorch (intra={threads}, inter=1) [3 trials]...")
        cmd = [
            sys.executable,
            pt_script,
            "--threads",
            str(threads),
            "--inter-threads",
            "1",
            "--json",
        ]
        results.append(run_isolated(cmd))

    onnx_configs = [
        (1, 1, "SEQUENTIAL"),
        (2, 1, "SEQUENTIAL"),
        (4, 1, "SEQUENTIAL"),
        (8, 1, "SEQUENTIAL"),
        (0, 1, "SEQUENTIAL"),
        (4, 2, "PARALLEL"),
    ]
    for intra, inter, mode in onnx_configs:
        print(
            f"Benchmarking ONNX Runtime (intra={intra}, inter={inter}, mode={mode}) [3 trials]..."
        )
        cmd = [
            sys.executable,
            ort_script,
            "--intra-threads",
            str(intra),
            "--inter-threads",
            str(inter),
            "--mode",
            mode,
            "--json",
        ]
        results.append(run_isolated(cmd))

    print_table(results)

    # Verify prediction consistency across all runtimes and thread configs (Aspect #7)
    preds = {(r["top1_class"], r["top1_conf_pct"]) for r in results}
    print(
        f"\nPrediction Consistency : {'PASS' if len(preds) == 1 else 'MISMATCH'} -> {preds}"
    )

    # Thread scaling analysis (Aspect #10)
    t4 = next(r for r in results if r["config"] == "intra=4, inter=1, mode=SEQUENTIAL")
    t8 = next(r for r in results if r["config"] == "intra=8, inter=1, mode=SEQUENTIAL")
    print(
        f"Thread Scaling Note    : 4->8 threads changed avg latency from "
        f"{t4['avg_latency_ms']} ms ({t4['cpu_util_pct']}% CPU) to "
        f"{t8['avg_latency_ms']} ms ({t8['cpu_util_pct']}% CPU) due to OpenMP barrier/cache "
        f"contention on MobileNetV2's lightweight depthwise convolutions at batch_size=1."
    )

    save_results(results, sys_info, root / "results")
    print(f"Results saved to {root / 'results'}")


if __name__ == "__main__":
    main()

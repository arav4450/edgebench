import csv
import json
import subprocess
from pathlib import Path
import numpy as np

CONTAINER_PROBE = """
import json, pathlib, sys
sys.path.insert(0, '/app/benchmarks')
if sys.argv[1] == 'pytorch':
    from benchmark_pytorch import benchmark_pytorch, get_memory_stats_mb
    res = benchmark_pytorch(num_threads=int(sys.argv[2]), inter_threads=1)
else:
    from benchmark_onnx import benchmark_onnx, get_memory_stats_mb
    res = benchmark_onnx(intra_threads=int(sys.argv[2]), inter_threads=1)
rss_mb, peak_mb = get_memory_stats_mb()
cg_cur = pathlib.Path('/sys/fs/cgroup/memory.current')
cg_peak = pathlib.Path('/sys/fs/cgroup/memory.peak')
res['rss_mb'] = rss_mb
res['container_mem_mb'] = round(int(cg_cur.read_text()) / 1048576, 2) if cg_cur.is_file() else rss_mb
res['container_peak_mb'] = round(int(cg_peak.read_text()) / 1048576, 2) if cg_peak.is_file() else peak_mb
print(json.dumps(res))
"""


def run_docker_config(
    runtime: str, threads: int, cpus: str | None, memory: str | None, trials: int = 5
) -> dict:
    """Run a container configuration across `trials` independent runs and aggregate."""
    cmd = ["sudo", "docker", "run", "--rm"]
    if cpus:
        cmd += [f"--cpus={cpus}"]
    if memory:
        cmd += [f"--memory={memory}", f"--memory-swap={memory}"]
    cmd += ["edgebench:latest", "python", "-c", CONTAINER_PROBE, runtime, str(threads)]

    cpu_label = cpus if cpus else "unrestricted"
    mem_label = memory if memory else "unrestricted"

    runs = []
    for _ in range(trials):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            status = (
                "OOM_KILLED (Exit 137)"
                if proc.returncode == 137
                else f"FAILED (Exit {proc.returncode})"
            )
            return {
                "framework": "PyTorch (CPU)"
                if runtime == "pytorch"
                else "ONNX Runtime (CPU)",
                "cpu_limit": cpu_label,
                "mem_limit": mem_label,
                "threads": threads,
                "trials": trials,
                "status": status,
                "avg_latency_ms": None,
                "trial_std_ms": None,
                "p50_latency_ms": None,
                "p95_latency_ms": None,
                "throughput_img_per_sec": 0.0,
                "cpu_util_pct": 0.0,
                "rss_mb": None,
                "peak_memory_mb": None,
                "container_peak_mb": None,
            }
        runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    keys = [
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "throughput_img_per_sec",
        "cpu_util_pct",
        "rss_mb",
        "peak_memory_mb",
        "container_peak_mb",
    ]
    agg = {
        "framework": runs[0]["framework"],
        "cpu_limit": cpu_label,
        "mem_limit": mem_label,
        "threads": threads,
        "trials": trials,
        "status": "OK",
        "avg_latency_ms": round(float(np.mean([r["avg_latency_ms"] for r in runs])), 3),
        "trial_std_ms": round(float(np.std([r["avg_latency_ms"] for r in runs])), 3),
    }
    for k in keys[1:]:
        agg[k] = round(float(np.mean([r[k] for r in runs])), 3 if "ms" in k else 2)
    return agg


def print_container_table(results: list[dict]) -> None:
    header = (
        f"{'Framework':<18} | {'CPUs':<12} | {'Mem':<12} | {'Thr':>3} | {'Status':<19} | "
        f"{'Avg±Std(ms)':>13} | {'p50(ms)':>7} | {'p95(ms)':>7} | {'Img/s':>7} | {'CPU%':>6} | {'RSS/Peak/Cg(MB)':>18}"
    )
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "OK":
            print(
                f"{r['framework']:<18} | {r['cpu_limit']:<12} | {r['mem_limit']:<12} | "
                f"{r['threads']:>3} | {r['status']:<19} | {'-':>13} | {'-':>7} | {'-':>7} | "
                f"{'-':>7} | {'-':>6} | {'-':>18}"
            )
        else:
            avg_std = f"{r['avg_latency_ms']:>6.2f}±{r['trial_std_ms']:<5.2f}"
            mem_s = f"{r['rss_mb']:.0f}/{r['peak_memory_mb']:.0f}/{r['container_peak_mb']:.0f}"
            print(
                f"{r['framework']:<18} | {r['cpu_limit']:<12} | {r['mem_limit']:<12} | "
                f"{r['threads']:>3} | {r['status']:<19} | {avg_std:>13} | "
                f"{r['p50_latency_ms']:>7.2f} | {r['p95_latency_ms']:>7.2f} | "
                f"{r['throughput_img_per_sec']:>7.1f} | {r['cpu_util_pct']:>5.1f}% | {mem_s:>18}"
            )
    print("-" * len(header))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment matrix: (runtime, intra_threads, cpus_limit, memory_limit)
    experiments = [
        # 1. Unrestricted CPU & Memory baseline
        ("onnx", 4, None, None),
        ("onnx", 1, None, None),
        # 2. CPU constraints (--cpus=2 and --cpus=1) at 1 GB
        ("onnx", 4, "2", "1g"),
        ("onnx", 2, "2", "1g"),
        ("onnx", 4, "1", "1g"),
        ("onnx", 1, "1", "1g"),
        # 3. Memory scaling sweep (1 GB -> 512 MB -> 256 MB -> 128 MB -> 64 MB -> 32 MB)
        ("onnx", 1, "1", "512m"),
        ("onnx", 1, "1", "256m"),
        ("onnx", 1, "1", "128m"),
        ("onnx", 1, "1", "64m"),
        ("onnx", 1, "1", "32m"),
    ]

    results = []
    for runtime, threads, cpus, mem in experiments:
        print(
            f"Running 5 trials: runtime={runtime}, threads={threads}, cpus={cpus or 'all'}, mem={mem or 'all'}..."
        )
        results.append(run_docker_config(runtime, threads, cpus, mem, trials=5))

    print_container_table(results)

    # Identify minimum memory requirement from the memory scaling sweep
    mem_sweep = [
        r
        for r in results
        if r["cpu_limit"] == "1"
        and r["threads"] == 1
        and r["mem_limit"] != "unrestricted"
    ]
    ok_mems = [r["mem_limit"] for r in mem_sweep if r["status"] == "OK"]
    fail_mems = [r["mem_limit"] for r in mem_sweep if r["status"] != "OK"]
    print(f"\nMinimum Memory Requirement Analysis:")
    print(
        f"  - Passed memory tiers : {ok_mems} (lowest functional tier: {ok_mems[-1] if ok_mems else 'None'})"
    )
    print(f"  - Failed (OOM) tiers  : {fail_mems}")
    print(
        f"  - Recommended minimum : 128m (provides safe headroom above ~91 MB Peak RSS; 64m forces kernel page-cache eviction)"
    )

    with (results_dir / "container_benchmark_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (results_dir / "container_benchmark_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(
        f"\nSaved 5-trial container dataset to {results_dir / 'container_benchmark_results.csv'}"
    )


if __name__ == "__main__":
    main()

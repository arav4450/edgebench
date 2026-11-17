import json
from pathlib import Path
from PIL import Image, ImageDraw


def draw_comparison_chart(results: list[dict], out_path: Path) -> None:
    """Render a side-by-side PNG bar chart of Latency (ms) and Peak Memory (MB)."""
    ok_runs = [r for r in results if r["status"] == "OK"]
    width, row_h = 1000, 52
    height = 120 + len(results) * row_h
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text(
        (20, 20),
        "Container Benchmark Comparison: Avg/p95 Latency (ms) & Peak Memory (MB)",
        fill=(20, 20, 20),
    )
    draw.text(
        (20, 48),
        "Blue = Avg Latency (ms) | Orange = p95 Latency (ms) | Green = Peak RAM (MB) | Red = OOM/Fail",
        fill=(90, 90, 90),
    )

    max_lat = max((r["p95_latency_ms"] for r in ok_runs), default=50.0)
    for idx, r in enumerate(results):
        y = 85 + idx * row_h
        label = f"{r['framework'][:4]} | cpu={r['cpu_limit']} | mem={r['mem_limit']} | t={r['threads']}"
        draw.text((20, y + 10), label, fill=(0, 0, 0))

        if r["status"] != "OK":
            draw.rectangle([330, y + 6, 620, y + 32], fill=(220, 53, 69))
            draw.text((340, y + 12), r["status"], fill=(255, 255, 255))
            continue

        # Latency bars (x: 330..630)
        avg_w = int((r["avg_latency_ms"] / max_lat) * 260)
        p95_w = int((r["p95_latency_ms"] / max_lat) * 260)
        draw.rectangle([330, y + 6, 330 + p95_w, y + 20], fill=(255, 159, 64))
        draw.rectangle([330, y + 22, 330 + avg_w, y + 36], fill=(54, 162, 235))
        draw.text(
            (600, y + 12),
            f"{r['avg_latency_ms']:.1f}/{r['p95_latency_ms']:.1f} ms",
            fill=(0, 0, 0),
        )

        # Memory bar (x: 720..960, scaled to 1024 MB)
        mem_w = min(220, int((r["peak_memory_mb"] / 1024.0) * 220))
        draw.rectangle([720, y + 10, 720 + mem_w, y + 30], fill=(40, 167, 69))
        draw.text(
            (725 + mem_w, y + 12), f"{r['peak_memory_mb']:.0f} MB", fill=(0, 0, 0)
        )

    img.save(out_path)


def analyze() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "results" / "container_benchmark_results.json"
    results = json.loads(data_path.read_text())

    base_ort = next(
        r
        for r in results
        if r["framework"].startswith("ONNX") and r["cpu_limit"] == "unrestricted"
    )
    print("\n=== Performance Degradation & Bottleneck Analysis ===")
    for r in results:
        if r["status"] != "OK":
            print(
                f"- [MEMORY BOTTLENECK] {r['framework']} (cpu={r['cpu_limit']}, mem={r['mem_limit']}): {r['status']}"
            )
        elif r["framework"].startswith("ONNX"):
            slowdown = r["avg_latency_ms"] / base_ort["avg_latency_ms"]
            print(
                f"- {r['framework']} (cpu={r['cpu_limit']}, mem={r['mem_limit']}, threads={r['threads']}): "
                f"{r['avg_latency_ms']:.2f} ms ({slowdown:.2f}x vs unrestricted), p95={r['p95_latency_ms']:.2f} ms, CPU={r['cpu_util_pct']}%"
            )

    chart_path = root / "results" / "container_performance_chart.png"
    draw_comparison_chart(results, chart_path)
    print(f"\nSaved visual comparison chart to: {chart_path}")


if __name__ == "__main__":
    analyze()

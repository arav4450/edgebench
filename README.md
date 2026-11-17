# EdgeBench: CPU Inference Benchmarking & Optimization

The objective of this project is to develop a reproducible benchmarking and optimization framework to evaluate deep learning inference performance (`MobileNetV2`) on resource-constrained CPU systems. We conducted three core experiments across isolated trials: (1) comparing native PyTorch CPU against ONNX Runtime across intra-op/inter-op thread configurations, (2) profiling containerized ONNX Runtime under Linux cgroup CPU (`--cpus=1`, `--cpus=2`) and memory (`1g` down to `32m`) constraints, and (3) applying static per-channel INT8 quantization calibrated and evaluated on the Imagenette dataset.

## Setup & Execution

```bash
# 1. Install dependencies & export MobileNetV2 to ONNX + INT8
uv sync
uv run src/export_model.py
uv run src/quantize_model.py

# 2. Run host CPU thread-scaling & INT8 quantization benchmarks
uv run benchmarks/run_benchmarks.py
uv run benchmarks/benchmark_quantization.py

# 3. Build Docker container & run CPU/memory constraint benchmarks
sudo docker build -t edgebench:latest .
sudo -v && uv run benchmarks/run_container_benchmarks.py
uv run benchmarks/analyze_container_results.py
```

## Key Results

- **ONNX Runtime vs. PyTorch Baseline:** Exporting MobileNetV2 to ONNX Runtime (`intra=4, inter=1`) reduced average CPU latency by **4.4×** (`10.36 ms` → `2.37 ms`), increased throughput to **422.1 img/s**, and cut process memory from **~742 MB** (PyTorch) down to **~95 MB** (ONNX Runtime).
- **Thread Scaling Ceiling:** Scaling intra-op threads from `1` to `4` improved ONNX Runtime latency nearly **3×** (`6.97 ms` → `2.37 ms`), while scaling further to `8` threads degraded latency to `3.59 ms` due to OpenMP synchronization and cache contention on lightweight depthwise convolutions at `batch_size=1`.
- **Container CPU & Memory Constraints:** Matching application thread count to the container `--cpus` quota eliminated CFS throttling spikes (`--cpus=1` with `threads=1` achieved `11.00 ms` avg / `14.26 ms` p95 vs. `40.36 ms` avg / `93.52 ms` p95 with `threads=4`), while memory scaling identified **128 MB** as the safe minimum container allocation (`64 MB` survived via page-cache eviction and `32 MB` triggered `OOM_KILLED / Exit 137`).
- **Static Per-Channel INT8 Quantization:** Quantizing to INT8 shrunk the model size by **3.6×** (`13.34 MB` → `3.72 MB`), reduced model RAM delta by **40%** (`42.9 MB` → `25.4 MB`), and nearly doubled 4-thread throughput (`256.5 img/s` → `495.0 img/s` at `2.02 ms` latency) while retaining **64.4% Top-1 accuracy** on Imagenette (within **1.0 percentage point** of the `65.4%` FP32 baseline, with **87.4%** prediction agreement).

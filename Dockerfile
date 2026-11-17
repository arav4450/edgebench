FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
# Copy pre-installed Python 3.11 (linux/x86_64) ONNX Runtime, NumPy, and Pillow from local .venv (0 network needed)
COPY .venv/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Copy ONNX model, sample image, inference script, and benchmarks
COPY models/ ./models/
COPY data/ ./data/
COPY src/onnx_inference.py ./src/onnx_inference.py
COPY benchmarks/benchmark_onnx.py ./benchmarks/benchmark_onnx.py

# Default command: verify ONNX inference inside container
CMD ["python", "src/onnx_inference.py"]
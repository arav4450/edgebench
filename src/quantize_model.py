import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision.datasets import Imagenette

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.benchmark_onnx import preprocess_image


class ImagenetteCalibrationReader(CalibrationDataReader):
    """Feeds calibration images from Imagenette into ONNX Runtime static quantization."""

    def __init__(self, image_paths: list[Path], input_name: str) -> None:
        self.image_paths = image_paths
        self.input_name = input_name
        self._iter = iter(self.image_paths)

    def get_next(self) -> dict | None:
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: preprocess_image(path)}


def get_calibration_images(data_root: Path, per_class: int = 10) -> list[Path]:
    """Download Imagenette (160px) if needed and sample `per_class` images per category."""
    extracted = data_root / "imagenette2-160"
    dataset = Imagenette(
        root=str(data_root),
        split="train",
        size="160px",
        download=not extracted.exists(),
    )
    counts: dict[int, int] = {}
    selected: list[Path] = []
    for img_path, label in dataset._samples:
        if counts.get(label, 0) < per_class:
            selected.append(Path(img_path))
            counts[label] = counts.get(label, 0) + 1
    return selected


def quantize_mobilenet_v2() -> None:
    root = Path(__file__).resolve().parent.parent
    fp32_path = root / "models" / "mobilenet_v2.onnx"
    prep_path = root / "models" / "mobilenet_v2_prep.onnx"
    int8_path = root / "models" / "mobilenet_v2_int8.onnx"
    data_root = root / "data" / "imagenette"

    print(
        "1. Preparing Imagenette calibration dataset (100 images across 10 classes)..."
    )
    calib_images = get_calibration_images(data_root, per_class=10)

    print("2. Running ONNX quantization pre-processing (shape inference & fusion)...")
    quant_pre_process(str(fp32_path), str(prep_path))

    session = ort.InferenceSession(str(prep_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = ImagenetteCalibrationReader(calib_images, input_name)

    print(
        "3. Applying Static INT8 Quantization (Per-Channel QDQ, QUInt8 activations / QInt8 weights)..."
    )
    quantize_static(
        model_input=str(prep_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prep_path.unlink(missing_ok=True)

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"\nQuantization complete -> {int8_path}")
    print(
        f"Model Size: FP32 = {fp32_mb:.2f} MB | INT8 = {int8_mb:.2f} MB ({fp32_mb / int8_mb:.2f}x smaller)"
    )


if __name__ == "__main__":
    quantize_mobilenet_v2()

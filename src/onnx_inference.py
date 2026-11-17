# ONNX Runtime inference

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from benchmarks.benchmark_onnx import preprocess_image


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def run_onnx_inference() -> None:
    project_root = Path(__file__).resolve().parent.parent
    onnx_model_path = project_root / "models" / "mobilenet_v2.onnx"
    image_folder = project_root / "data" / "sample_images"

    session = ort.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p
        for p in image_folder.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    )

    for image_path in image_paths:
        input_batch = preprocess_image(image_path)
        outputs = session.run([output_name], {input_name: input_batch})
        probabilities = softmax(outputs[0][0])
        top_catid = int(np.argmax(probabilities))
        top_prob = float(probabilities[top_catid])
        category_name = "Egyptian cat" if top_catid == 285 else f"class_{top_catid}"
        print(
            f"{image_path.name} -> Predicted Class: {category_name} ({top_prob * 100:.2f}%)"
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run_onnx_inference()

# Export PyTorch MobileNetV2 model to ONNX format

from pathlib import Path

import onnx
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def export_to_onnx() -> Path:
    # 1. Load the pretrained MobileNetV2 model and set to evaluation mode
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()

    # 2. Define output directory and ONNX file path
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = models_dir / "mobilenet_v2.onnx"

    # 3. Create dummy input tensor matching ImageNet input shape (B x C x H x W)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # 4. Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    # 5. Verify the exported ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print(f"Model successfully exported and verified at: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    export_to_onnx()

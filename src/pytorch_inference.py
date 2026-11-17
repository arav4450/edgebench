# pytorch inference

from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# 1. Load the pretrained MobileNetV2 model with default weights (ImageNet)
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# Set the model to evaluation mode (freezes dropout/batchnorm layers)
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Get the specific image transformation required by this model version
# This handles resizing (256), central crop (224), and ImageNet normalization
preprocess = weights.transforms()

# 3. Point to the sample images folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
image_folder = PROJECT_ROOT / "data" / "sample_images"

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted(
    p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions
)

if not image_paths:
    raise FileNotFoundError(f"No valid images found in {image_folder}")

# 4. Perform inference on each image in the folder
with torch.no_grad():
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(img)

        # Add a batch dimension (B x C x H x W) and move to the device
        input_batch = input_tensor.unsqueeze(0).to(device)

        output = model(input_batch)

        # 5. Extract probabilities and predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get the top predicted class index and score
        top_prob, top_catid = torch.max(probabilities, 0)

        # 6. Print the human-readable class name using the weight categories
        category_name = weights.meta["categories"][top_catid.item()]
        print(f"{image_path.name} -> Predicted Class: {category_name} ({top_prob.item() * 100:.2f}%)")


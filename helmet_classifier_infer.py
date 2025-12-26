import torch
from PIL import Image
import os
import argparse
from utils.labels import get_labels
from models.helmet import get_helmet_model, get_helmet_infer_transform

# ------------------
# Config
# ------------------
WEIGHTS_PATH = "weights/helmet_resnet18_best.pth"
IMG_SIZE = 224
CLASSES = get_labels()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# Model
# ------------------
model = get_helmet_model(WEIGHTS_PATH, DEVICE)

# ------------------
# Transforms
# ------------------
transform = get_helmet_infer_transform(IMG_SIZE)

# ------------------
# Inference function
# ------------------
def infer_image(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    return CLASSES[pred], probs[pred].item()

# ------------------
# Main
# ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to image")
    parser.add_argument("--dir", type=str, help="Path to image folder")
    args = parser.parse_args()

    if args.img:
        label, conf = infer_image(args.img)
        print(f"{args.img} → {label} ({conf:.3f})")

    elif args.dir:
        for file in os.listdir(args.dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(args.dir, file)
                label, conf = infer_image(path)
                print(f"{file} → {label} ({conf:.3f})")

    else:
        print("Please specify --img or --dir")

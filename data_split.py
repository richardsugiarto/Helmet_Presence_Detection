import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from utils.labels import get_labels

# -----------------------------
# Config
# -----------------------------
SOURCE_DIR = "data/helmet/all"
TARGET_DIR = "data/helmet"
VAL_RATIO = 0.2
RANDOM_STATE = 42

classes = get_labels()

# -----------------------------
# Create target folders
# -----------------------------
for split in ["train", "val"]:
    for cls in classes:
        folder = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(folder, exist_ok=True)

# -----------------------------
# Split images and copy
# -----------------------------
for cls in classes:
    cls_source = os.path.join(SOURCE_DIR, cls)
    images = glob.glob(os.path.join(cls_source, "*.jpg"))
    
    train_imgs, val_imgs = train_test_split(
        images, test_size=VAL_RATIO, random_state=RANDOM_STATE
    )
    
    # Copy train images
    for img_path in train_imgs:
        dst = os.path.join(TARGET_DIR, "train", cls, os.path.basename(img_path))
        shutil.copy(img_path, dst)
    
    # Copy val images
    for img_path in val_imgs:
        dst = os.path.join(TARGET_DIR, "val", cls, os.path.basename(img_path))
        shutil.copy(img_path, dst)

print("Dataset split completed!")

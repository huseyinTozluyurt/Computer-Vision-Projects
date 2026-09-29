import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_IMAGE_DIR = PROJECT_DIR / "dataset/images/raw"
RAW_LABEL_DIR = PROJECT_DIR / "dataset/labels/raw"

TRAIN_IMAGE_DIR = PROJECT_DIR / "dataset/images/train"
VAL_IMAGE_DIR = PROJECT_DIR / "dataset/images/val"

TRAIN_LABEL_DIR = PROJECT_DIR / "dataset/labels/train"
VAL_LABEL_DIR = PROJECT_DIR / "dataset/labels/val"

for folder in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Optional cleanup: prevents old files from mixing with new split
for folder in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
]:
    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(RAW_IMAGE_DIR.glob(ext))

if not image_paths:
    print("No images found in dataset/images/raw/")
    exit()

random.seed(42)
random.shuffle(image_paths)

train_ratio = 0.8
train_count = int(len(image_paths) * train_ratio)

train_images = image_paths[:train_count]
val_images = image_paths[train_count:]


def copy_pair(image_path, image_output_dir, label_output_dir):
    label_path = RAW_LABEL_DIR / f"{image_path.stem}.txt"

    shutil.copy2(image_path, image_output_dir / image_path.name)

    if label_path.exists():
        shutil.copy2(label_path, label_output_dir / label_path.name)
    else:
        empty_label_path = label_output_dir / f"{image_path.stem}.txt"
        empty_label_path.touch()
        print(f"Warning: no label found for {image_path.name}. Created empty label file.")


for image_path in train_images:
    copy_pair(image_path, TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)

for image_path in val_images:
    copy_pair(image_path, VAL_IMAGE_DIR, VAL_LABEL_DIR)

print("Dataset split completed.")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
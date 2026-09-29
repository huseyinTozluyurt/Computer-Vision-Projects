import random
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_IMAGE_DIR = PROJECT_DIR / "dataset/images/train"
TRAIN_LABEL_DIR = PROJECT_DIR / "dataset/labels/train"

VAL_IMAGE_DIR = PROJECT_DIR / "dataset/images/val"
VAL_LABEL_DIR = PROJECT_DIR / "dataset/labels/val"

VAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABEL_DIR.mkdir(parents=True, exist_ok=True)

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(TRAIN_IMAGE_DIR.glob(ext))

if not image_paths:
    print("No images found in dataset/images/train/")
    exit()

random.seed(42)
random.shuffle(image_paths)

val_ratio = 0.2
val_count = max(1, int(len(image_paths) * val_ratio))

val_images = image_paths[:val_count]

print(f"Total train images before split: {len(image_paths)}")
print(f"Moving {len(val_images)} images to validation set.")

for image_path in val_images:
    label_path = TRAIN_LABEL_DIR / f"{image_path.stem}.txt"

    target_image_path = VAL_IMAGE_DIR / image_path.name
    target_label_path = VAL_LABEL_DIR / f"{image_path.stem}.txt"

    shutil.move(str(image_path), str(target_image_path))

    if label_path.exists():
        shutil.move(str(label_path), str(target_label_path))
    else:
        target_label_path.touch()
        print(f"Warning: label missing for {image_path.name}. Empty label created.")

print("Split completed.")

print("\nFinal counts:")
print(f"Train images: {len(list(TRAIN_IMAGE_DIR.glob('*')))}")
print(f"Train labels: {len(list(TRAIN_LABEL_DIR.glob('*.txt')))}")
print(f"Val images: {len(list(VAL_IMAGE_DIR.glob('*')))}")
print(f"Val labels: {len(list(VAL_LABEL_DIR.glob('*.txt')))}")

from ultralytics import YOLO
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]

model = YOLO("yolov8s.pt")

# COCO class ids:
# person = 0
# bicycle = 1
# car = 2
# traffic light = 9
# cat = 15

# Your custom class ids:
# 0 = car
# 1 = bicycle
# 2 = person
# 3 = cat
# 4 = traffic_light

COCO_TO_CUSTOM = {
    2: 0,    # car -> car
    1: 1,    # bicycle -> bicycle
    0: 2,    # person -> person
    15: 3,   # cat -> cat
    9: 4,    # traffic light -> traffic_light
}

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def auto_label_folder(image_dir, label_dir):
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_dir.glob(ext))

    if not image_paths:
        print(f"No images found in: {image_dir}")
        return

    print(f"\nFound {len(image_paths)} images in {image_dir}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]

        results = model.predict(
            source=image,
            conf=0.35,
            verbose=False
        )

        label_lines = []

        for result in results:
            for box in result.boxes:
                coco_class_id = int(box.cls[0])

                if coco_class_id not in COCO_TO_CUSTOM:
                    continue

                custom_class_id = COCO_TO_CUSTOM[coco_class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                label_line = (
                    f"{custom_class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_width:.6f} "
                    f"{box_height:.6f}"
                )

                label_lines.append(label_line)

        label_path = label_dir / f"{image_path.stem}.txt"

        with open(label_path, "w", encoding="utf-8") as f:
            for line in label_lines:
                f.write(line + "\n")

        print(
            f"Auto-labeled: {image_path.name} -> {label_path.name} "
            f"| objects: {len(label_lines)}"
        )


train_image_dir = PROJECT_DIR / "dataset/images/train"
train_label_dir = PROJECT_DIR / "dataset/labels/train"

val_image_dir = PROJECT_DIR / "dataset/images/val"
val_label_dir = PROJECT_DIR / "dataset/labels/val"

auto_label_folder(train_image_dir, train_label_dir)
auto_label_folder(val_image_dir, val_label_dir)

print("\nAuto-labeling completed.")
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_IMAGE_DIR = PROJECT_DIR / "dataset/images/raw"
LABEL_OUTPUT_DIR = PROJECT_DIR / "dataset/labels/train"

CLASSES_PATH = PROJECT_DIR / "classes.txt"

LABEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f.readlines() if line.strip()]

print("Classes:")
for i, cls in enumerate(CLASSES):
    print(f"{i}: {cls}")

image_paths = list(RAW_IMAGE_DIR.glob("*.jpg")) + list(RAW_IMAGE_DIR.glob("*.png")) + list(RAW_IMAGE_DIR.glob("*.jpeg"))

if not image_paths:
    print("No images found in dataset/images/raw/")
    exit()

current_boxes = []
drawing = False
start_x, start_y = -1, -1
selected_class_id = 0
current_image = None
display_image = None


def yolo_format(box, image_width, image_height):
    class_id, x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height
    width = abs(x2 - x1) / image_width
    height = abs(y2 - y1) / image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def draw_boxes():
    global display_image
    display_image = current_image.copy()

    for box in current_boxes:
        class_id, x1, y1, x2, y2 = box
        label = CLASSES[class_id]

        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        display_image,
        f"Selected class: {selected_class_id} - {CLASSES[selected_class_id]}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display_image,
        "Keys: 0-4 class | s save | u undo | n next | q quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        x1, x2 = sorted([start_x, end_x])
        y1, y2 = sorted([start_y, end_y])

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            current_boxes.append((selected_class_id, x1, y1, x2, y2))
            draw_boxes()


def save_labels(image_path, image_width, image_height):
    label_path = LABEL_OUTPUT_DIR / f"{image_path.stem}.txt"

    with open(label_path, "w", encoding="utf-8") as f:
        for box in current_boxes:
            f.write(yolo_format(box, image_width, image_height) + "\n")

    print(f"Saved labels: {label_path}")


cv2.namedWindow("Manual YOLO Labeler")
cv2.setMouseCallback("Manual YOLO Labeler", mouse_callback)

for image_path in image_paths:
    current_image = cv2.imread(str(image_path))

    if current_image is None:
        print(f"Could not read image: {image_path}")
        continue

    height, width = current_image.shape[:2]
    current_boxes = []
    draw_boxes()

    print(f"\nLabeling image: {image_path.name}")

    while True:
        cv2.imshow("Manual YOLO Labeler", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(i)) for i in range(len(CLASSES))]:
            selected_class_id = int(chr(key))
            print(f"Selected class: {selected_class_id} - {CLASSES[selected_class_id]}")
            draw_boxes()

        elif key == ord("s"):
            save_labels(image_path, width, height)

        elif key == ord("u"):
            if current_boxes:
                removed = current_boxes.pop()
                print(f"Removed box: {removed}")
                draw_boxes()

        elif key == ord("n"):
            save_labels(image_path, width, height)
            break

        elif key == ord("q"):
            save_labels(image_path, width, height)
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("Labeling completed.")
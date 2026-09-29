from ultralytics import YOLO
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "runs/detect/custom_car_bicycle_person_cat_traffic_light/weights/best.pt"

CLASS_NAMES = {
    0: "car",
    1: "bicycle",
    2: "person",
    3: "cat",
    4: "traffic_light",
}

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing camera source from 0 to 1.")

# Request larger camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create resizable OpenCV window
WINDOW_NAME = "Custom YOLO Detection with Counter"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    object_counts = {
        "car": 0,
        "bicycle": 0,
        "person": 0,
        "cat": 0,
        "traffic_light": 0,
    }

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id in CLASS_NAMES:
            class_name = CLASS_NAMES[class_id]
            object_counts[class_name] += 1

    annotated_frame = result.plot()

    # Draw counter panel
    panel_x = 20
    panel_y = 40
    line_height = 30

    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (330, 210),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        annotated_frame,
        "Object Counter",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y = panel_y + line_height

    for class_name, count in object_counts.items():
        text = f"{class_name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y += line_height

    # Display at larger size
    display_frame = cv2.resize(annotated_frame, (1280, 720))

    cv2.imshow(WINDOW_NAME, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
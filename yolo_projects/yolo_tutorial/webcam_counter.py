import cv2
from collections import defaultdict
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.predict(
            source=frame,
            conf=0.25,
            verbose=False
        )

        result = results[0]
        counts = defaultdict(int)

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            counts[class_name] += 1

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            label = f"{class_name}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        y_offset = 30

        for class_name, count in counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            y_offset += 30

        cv2.imshow("YOLO Object Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
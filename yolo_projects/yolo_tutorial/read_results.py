from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="images/",
    conf=0.25,
    save=True
)

# results is a list. For one image, use results[0]
result = results[0]

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0]

    class_name = model.names[class_id]

    print("Class:", class_name)
    print("Confidence:", confidence)
    print("Box:", float(x1), float(y1), float(x2), float(y2))
    print("-------------------------")
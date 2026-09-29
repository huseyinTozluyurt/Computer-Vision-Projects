from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run prediction on image
results = model.predict(
    source="images/",
    conf=0.12,
    save=True
)

print("Image prediction completed.")
print(results)
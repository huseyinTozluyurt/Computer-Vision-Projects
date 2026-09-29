from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

folders = [
    "dataset/images/raw",
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val",
]

for folder in folders:
    path = PROJECT_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {path}")

classes = [
    "car",
    "bicycle",
    "person",
    "cat",
    "traffic light",
]

classes_path = PROJECT_DIR / "classes.txt"

with open(classes_path, "w", encoding="utf-8") as f:
    for cls in classes:
        f.write(cls + "\n")

data_yaml = PROJECT_DIR / "data.yaml"

with open(data_yaml, "w", encoding="utf-8") as f:
    f.write("""path: dataset
train: images/train
val: images/val

names:
  0: car
  1: bicycle
  2: person
  3: cat
  4: traffic light
""")

print("classes.txt and data.yaml created successfully.")
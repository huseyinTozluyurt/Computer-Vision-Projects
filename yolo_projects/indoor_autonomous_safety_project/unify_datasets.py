import os
import shutil
from pathlib import Path

# 1. Define your explicit master labels (Separate classes for ROS2 navigation routing)
MASTER_CLASSES = {
    "cat": 0,
    "dog": 1,
    "human": 2,
    "shoe": 3,
    "trouser": 4
}

# 2. Map where the datasets are and what they internally call their elements
DATASETS_CONFIG = [
    {"folder": "cat", "mapping": {0: "cat"}},
    {"folder": "dog", "mapping": {0: "dog"}},
    {"folder": "human_dataset", "mapping": {0: "human"}},
    {"folder": "shoe_dataset", "mapping": {0: "shoe"}},
    {"folder": "trouser", "mapping": {0: "trouser"}}
]

BASE_DIR = Path("/home/huseyin/opencv_env/yolo_projects/indoor_autonomous_safety_project/indoor_autonomous_safety_datasets")
OUTPUT_DIR = Path("/home/huseyin/opencv_env/yolo_projects/indoor_autonomous_safety_project/unified_safety_dataset")

def process_split(split_name):
    # Mapping Roboflow 'valid' folder to standard 'val' format expected by YOLO
    source_split = "valid" if split_name == "val" else split_name
    
    out_img_dir = OUTPUT_DIR / split_name / "images"
    out_lbl_dir = OUTPUT_DIR / split_name / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS_CONFIG:
        ds_path = BASE_DIR / ds["folder"]
        img_src = ds_path / source_split / "images"
        lbl_src = ds_path / source_split / "labels"
        
        if not img_src.exists() or not lbl_src.exists():
            print(f"⚠️ Split '{source_split}' not found in {ds['folder']}. Skipping...")
            continue

        print(f"Processing {split_name} from: {ds['folder']}...")

        for img_file in img_src.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                lbl_file = lbl_src / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    continue

                # Prepend folder name to guarantee unique file names across merges
                unique_name = f"{ds['folder']}_{img_file.name}"
                unique_lbl_name = f"{ds['folder']}_{lbl_file.name}"

                # Copy image
                shutil.copy(img_file, out_img_dir / unique_name)

                # Process labels and offset class indexes
                new_lines = []
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        old_cls_id = int(parts[0])
                        if old_cls_id in ds["mapping"]:
                            target_class_name = ds["mapping"][old_cls_id]
                            new_cls_id = MASTER_CLASSES[target_class_name]
                            
                            parts[0] = str(new_cls_id)
                            new_lines.append(" ".join(parts))

                with open(out_lbl_dir / unique_lbl_name, "w") as f:
                    f.write("\n".join(new_lines) + "\n")

# Run mapping for all splits
for split in ["train", "val", "test"]:
    process_split(split)

print(f"\n🎉 Multi-class dataset successfully created at: {OUTPUT_DIR}")
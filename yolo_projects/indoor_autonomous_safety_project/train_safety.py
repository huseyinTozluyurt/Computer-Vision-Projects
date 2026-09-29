from ultralytics import YOLO

def main():
    # Load YOLOv8 Nano base weights
    model = YOLO("yolov8n.pt") 

    # Begin fine-tuning with specific hardware flags for the GTX 1650
    model.train(
        data="/home/huseyin/opencv_env/yolo_projects/indoor_autonomous_safety_project/unified_safety_dataset/data.yaml",
        epochs=50,         
        imgsz=640,         
        batch=8,           # Lowered from 16 to 8 to fit safely in your 4GB VRAM
        device=0,          
        amp=False,         # <--- CRITICAL FIX: Explicitly turns off Mixed Precision to prevent the crash
        workers=2          # Lowered to 2 to prevent thread-locking memory management issues
    )

    print("🎉 Training successfully completed!")

if __name__ == '__main__':
    main()
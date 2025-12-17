from ultralytics import YOLO

# Load model dasar
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    # Proses melatih AI
    model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        device='cpu'  # Ganti jadi 0 jika laptop Anda punya NVIDIA GPU
    )
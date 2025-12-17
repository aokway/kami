from ultralytics import YOLO

model = YOLO(r'E:\PINDAIMOBIL_yolov8\runs\detect\train2\weights\best.pt')

# TAMBAHKAN BARIS INI untuk mengubah nama label secara paksa
model.names[2] = 'Mobil' 

results = model.predict(
    source='dataset/train/images', 
    save=True,      
    classes=[2],    
    conf=0.5
)
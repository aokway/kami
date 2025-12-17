from ultralytics import YOLO
import os

# 1. Gunakan model YOLOv8 yang sudah pintar mendeteksi mobil umum
model = YOLO('yolov8n.pt') 

# 2. Tentukan folder gambar Anda
path_gambar = 'dataset/train/images'

# 3. Jalankan deteksi dan simpan hasilnya sebagai label .txt
results = model.predict(source=path_gambar, save_txt=True, conf=0.5)

print("Proses selesai! Cek folder: runs/detect/predict/labels")
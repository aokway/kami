from ultralytics import YOLO
import cv2

# 1. Load model hasil training Anda
model = YOLO(r'E:\PINDAIMOBIL_yolov8\runs\detect\train2\weights\best.pt')

# 2. PAKSA LABEL: Ganti 'car' menjadi nama mobil spesifik Anda
# Sesuaikan 'Avanza/Xenia' dengan nama yang Anda inginkan
model.names[2] = "Mobil Avanza/Xenia" 

# 3. Coba buka kamera (Index 0, 1, atau 2)
# Jika Logitech biasanya di index 1 jika laptop ada kamera bawaan
cap = cv2.VideoCapture(1
) 

if not cap.isOpened():
    print("Gagal membuka kamera index 0, mencoba index 1...")
    cap = cv2.VideoCapture(1)

print("Kamera terbuka! Tekan 'q' untuk berhenti.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil gambar dari kamera.")
        break

    # Jalankan deteksi
    # classes=[2] karena mobil Anda di ID 2
    results = model.predict(frame, conf=0.7, classes=[2], verbose=False)

    # Gambar kotak dan label baru ke frame
    annotated_frame = results[0].plot()

    # Tampilkan di jendela
    cv2.imshow("Logitech Detection - Tekan Q untuk Keluar", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
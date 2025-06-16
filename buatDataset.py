# nama file: buatDataset.py
import os
import pickle
import mediapipe as mp
import cv2

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
# Konfigurasi untuk mendeteksi hingga 2 tangan dengan tingkat kepercayaan minimal 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Lokasi folder dataset
DATA_DIR = './data'

# List untuk menampung data fitur dan label
data = []
labels = []

print("Memulai proses pembuatan dataset...")

# Iterasi melalui setiap folder (A, B, C, ...)
for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    # Lewati jika bukan sebuah direktori/folder
    if not os.path.isdir(dir_path):
        continue

    print(f"Memproses folder: {dir_}")

    # Iterasi melalui setiap gambar di dalam folder
    for img_path in os.listdir(dir_path):
        # Hanya proses file gambar
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        data_aux = [] # List sementara untuk landmark satu gambar
        x_coords = [] # List untuk koordinat x
        y_coords = [] # List untuk koordinat y

        img_full_path = os.path.join(dir_path, img_path)
        
        # Baca gambar menggunakan OpenCV
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Peringatan: Gagal membaca gambar {img_full_path}")
            continue

        # Ubah gambar dari BGR ke RGB karena MediaPipe menggunakan format RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Proses gambar untuk mendeteksi tangan
        results = hands.process(img_rgb)

        # Jika tangan terdeteksi
        if results.multi_hand_landmarks:
            # Kumpulkan semua koordinat untuk mencari nilai minimum
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)

            # Cari titik referensi (koordinat x dan y terkecil)
            min_x, min_y = min(x_coords), min(y_coords)

            # Normalisasi koordinat dan tambahkan ke data_aux
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min_x, lm.y - min_y])

            # Jika hanya 1 tangan yang terdeteksi, tambahkan padding (nilai 0)
            # untuk tangan kedua agar semua data memiliki panjang yang sama (84).
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * 42) # 21 landmark * 2 koordinat = 42

            # Pastikan panjang fitur adalah 84 (untuk 2 tangan)
            if len(data_aux) == 84:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Peringatan: Panjang fitur tidak konsisten untuk {img_full_path}: {len(data_aux)}")

print(f"\nDataset selesai diproses: {len(data)} sampel berhasil dikumpulkan.")

# Simpan data dan label ke dalam file pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset berhasil disimpan ke dalam file 'data.pickle'")
# nama file: mulaiDeteksi_NN.py
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import random
from tensorflow.keras.models import load_model

class PengenalBahasaIsyaratNN:
    def __init__(self):
        try:
            # Muat model Jaringan Saraf Tiruan yang sudah dilatih
            self.model = load_model('model_nn.h5')
            # Muat label encoder
            with open('label_encoder.pickle', 'rb') as f:
                self.encoder = pickle.load(f)
            print("✓ Model dan Label Encoder berhasil dimuat.")
        except Exception as e:
            print(f"❌ Terjadi kesalahan saat memuat model atau encoder: {e}")
            return

        # Inisialisasi webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )

        # Variabel untuk manajemen prediksi dan suara
        self.prediksi_saat_ini = ""
        self.waktu_suara_terakhir = 0
        self.gambar_terakhir = None

        # Inisialisasi Pygame untuk memutar suara
        pygame.mixer.init()

    def putar_suara(self, huruf):
        try:
            sound_path = f'sounds/{huruf.upper()}.wav'
            if os.path.exists(sound_path):
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
            else:
                 print(f"🔊 File suara untuk '{huruf}' tidak ditemukan, dilewati.")
        except Exception as e:
            print(f"❌ Gagal memutar suara untuk '{huruf}': {e}")

    def muat_gambar_acak(self, huruf):
        folder_path = f'data/{huruf.upper()}'
        if not os.path.exists(folder_path): return None
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files: return None
        
        image_path = os.path.join(folder_path, random.choice(files))
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (400, 600))
        return image

    def proses_frame(self, frame):
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_coords = []
            y_coords = []

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)

            # Gambar landmark pada frame
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
                )

            # Normalisasi data landmark
            min_x, min_y = min(x_coords), min(y_coords)
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min_x, lm.y - min_y])

            # Tambahkan padding jika hanya 1 tangan
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * 42)

            if len(data_aux) == 84:
                # Lakukan prediksi dengan model NN
                prediction_probs = self.model.predict(np.array([data_aux]))
                skor_kepercayaan = np.max(prediction_probs)
                indeks_prediksi = np.argmax(prediction_probs)
                huruf_prediksi = self.encoder.inverse_transform([indeks_prediksi])[0]

                waktu_sekarang = time.time()
                # Putar suara jika kepercayaan tinggi dan sudah lewat 1 detik dari suara terakhir
                if skor_kepercayaan >= 0.7 and waktu_sekarang - self.waktu_suara_terakhir >= 1.0:
                    self.putar_suara(huruf_prediksi)
                    self.waktu_suara_terakhir = waktu_sekarang
                    self.gambar_terakhir = self.muat_gambar_acak(huruf_prediksi)

                self.prediksi_saat_ini = huruf_prediksi

                # Buat kotak pembatas (bounding box) di sekitar tangan
                x1, y1 = int(min(x_coords) * W) - 20, int(min(y_coords) * H) - 20
                x2, y2 = int(max(x_coords) * W) + 20, int(max(y_coords) * H) + 20
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f"{huruf_prediksi} ({skor_kepercayaan:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            self.prediksi_saat_ini = ""

        return frame

    def gambar_ui(self, frame):
        cv2.putText(frame, "Pengenalan Bahasa Isyarat (NN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        teks_tampilan = self.prediksi_saat_ini if self.prediksi_saat_ini else "Tangan tidak terdeteksi"
        cv2.putText(frame, f"Terdeteksi: {teks_tampilan}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Tekan 'q' untuk keluar", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def jalankan(self):
        print("🚀 Memulai Pengenalan Bahasa Isyarat dengan Jaringan Saraf Tiruan...")

        if not hasattr(self, 'model'):
            print("Model tidak berhasil dimuat. Program berhenti.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # Balik frame secara horizontal
            frame_hasil = self.proses_frame(frame)
            self.gambar_ui(frame_hasil)

            cv2.imshow('Kamera - Pengenalan Bahasa Isyarat', frame_hasil)

            # Tampilkan gambar referensi jika ada
            if self.gambar_terakhir is not None:
                cv2.imshow('Referensi Isyarat', self.gambar_terakhir)
            
            # Hentikan program jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    recognizer = PengenalBahasaIsyaratNN()
    recognizer.jalankan()
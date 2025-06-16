# nama file: latihModel_NN.py
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Muat dataset dari file pickle
print("Memuat dataset...")
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Bentuk data: {data.shape}")
print(f"Jumlah sampel: {len(labels)}")

# 2. Lakukan Encoding pada Label
# Jaringan saraf tiruan memerlukan input target berupa angka, bukan teks ('A', 'B', ...).
# Ubah label teks menjadi angka (0, 1, 2, ...).
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
# Ubah label angka menjadi format one-hot encoding (categorical).
labels_categorical = to_categorical(labels_encoded)

# Simpan encoder untuk digunakan nanti saat proses deteksi
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

jumlah_kelas = len(encoder.classes_)
print(f"Jumlah kelas: {jumlah_kelas}")
print(f"Daftar Kelas: {list(encoder.classes_)}")

# 3. Bagi dataset menjadi data latih (train) dan data uji (test)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, shuffle=True, stratify=labels_categorical, random_state=42
)

print(f"Jumlah sampel latih: {len(x_train)}")
print(f"Jumlah sampel uji: {len(x_test)}")

# 4. Bangun Arsitektur Model Jaringan Saraf Tiruan (MLP)
model = Sequential([
    # Layer input, dengan bentuk input sesuai jumlah fitur (84)
    Dense(256, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5), # Mencegah overfitting
    Dense(128, activation='relu'),
    Dropout(0.5), # Mencegah overfitting
    Dense(64, activation='relu'),
    # Layer output, jumlah neuron sama dengan jumlah kelas, dengan aktivasi softmax untuk klasifikasi
    Dense(jumlah_kelas, activation='softmax')
])

model.summary()

# 5. Kompilasi Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definisikan callbacks untuk optimasi pelatihan
# EarlyStopping: menghentikan pelatihan jika tidak ada peningkatan akurasi validasi.
# ModelCheckpoint: menyimpan hanya model terbaik selama pelatihan.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('model_nn.h5', save_best_only=True, monitor='val_accuracy')

# 6. Latih Model
print("\nMemulai pelatihan model Jaringan Saraf Tiruan...")
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

# 7. Evaluasi Model
print("\nMengevaluasi model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f'\nAkurasi pada data uji: {accuracy * 100:.2f}%')
print(f'Loss pada data uji: {loss:.4f}')

print("\nModel berhasil disimpan sebagai 'model_nn.h5'")
print("Label encoder berhasil disimpan sebagai 'label_encoder.pickle'")
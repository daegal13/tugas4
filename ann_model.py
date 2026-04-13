import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("produksi_padi_data.csv")

# 🔥 WAJIB: GROUP BY TAHUN
df = df.groupby('tahun')['produksi_padi'].sum().reset_index()

print(df.head())

# =========================
# VISUALISASI DATA ASLI
# =========================
plt.figure()
plt.scatter(df['tahun'], df['produksi_padi'])
plt.title("Data Produksi Padi per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Produksi Padi")
plt.grid()
plt.show()

# =========================
# PREPROCESSING
# =========================
X = df[['tahun']].values
Y = df['produksi_padi'].values

# Normalisasi
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(np.hstack((X, Y.reshape(-1,1))))

X_scaled = data_scaled[:, 0].reshape(-1,1)
Y_scaled = data_scaled[:, 1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# =========================
# MODEL ANN
# =========================
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# =========================
# TRAINING
# =========================
history = model.fit(
    X_train, Y_train,
    epochs=200,
    validation_data=(X_test, Y_test),
    verbose=1
)

# =========================
# VISUALISASI LOSS TRAINING
# =========================
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Grafik Loss Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# =========================
# PREDIKSI
# =========================
Y_pred = model.predict(X_test)

# =========================
# VISUALISASI HASIL PREDIKSI
# =========================
plt.figure()
plt.scatter(X_test, Y_test, label="Data Aktual")
plt.scatter(X_test, Y_pred, label="Prediksi ANN")

plt.title("Perbandingan Data Aktual vs Prediksi")
plt.xlabel("Tahun (Scaled)")
plt.ylabel("Produksi (Scaled)")
plt.legend()
plt.grid()
plt.show()

# =========================
# SIMPAN MODEL
# =========================
model.save("model/model.h5")

print("✅ Model berhasil dibuat & grafik berhasil ditampilkan!")
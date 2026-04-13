from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

data = None
model = None
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


# ==========================
# HOME
# ==========================
@app.route('/')
def index():
    return render_template("index.html")


# ==========================
# UPLOAD CSV
# ==========================
@app.route('/upload', methods=['POST'])
def upload():
    global data

    file = request.files['file']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        df = pd.read_csv(path)

        # 🔥 Ambil kolom yang benar dari dataset kamu
        df_group = df.groupby('tahun')['produksi_padi'].sum().reset_index()

        data = df_group

        print("Data siap:")
        print(data.head())

        return render_template("index.html", message="✅ Dataset berhasil diupload!")

    return redirect(url_for('index'))


# ==========================
# TRAIN MODEL
# ==========================
def train_model():
    global model, data

    X = data[['tahun']].values
    y = data[['produksi_padi']].values

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = Sequential()
    model.add(Dense(16, input_dim=1, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_scaled, y_scaled, epochs=300, verbose=0)


# ==========================
# PREDICT
# ==========================
@app.route('/predict', methods=['POST'])
def predict():
    global data, model

    if data is None:
        return render_template("index.html", message="⚠️ Upload dataset dulu!")

    tahun = int(request.form['tahun'])

    # TRAIN MODEL
    train_model()

    # PREDIKSI
    tahun_scaled = scaler_x.transform([[tahun]])
    pred_scaled = model.predict(tahun_scaled)

    prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
    prediction = float(prediction)  # 🔥 FIX float32

    prediction = round(prediction, 2)

    # ======================
    # DATA UNTUK CHART
    # ======================
    tahun_list = [int(x) for x in data['tahun']]
    produksi_list = [float(x) for x in data['produksi_padi']]

    # Tambahkan hasil prediksi
    tahun_list.append(int(tahun))
    produksi_list.append(float(prediction))

    # ======================
    # GRAFIK PNG
    # ======================
    plt.figure()
    plt.plot(data['tahun'], data['produksi_padi'], marker='o')
    plt.scatter(tahun, prediction)
    plt.title("Prediksi Produksi Padi")
    plt.xlabel("Tahun")
    plt.ylabel("Produksi")
    plt.savefig(os.path.join(STATIC_FOLDER, 'grafik.png'))
    plt.close()

    return render_template(
        "index.html",
        prediction=prediction,
        tahun=tahun,
        labels=tahun_list,
        values=produksi_list
    )


# ==========================
# RUN
# ==========================
if __name__ == '__main__':
    app.run(debug=True)
import os
# Menyembunyikan pesan log TensorFlow level INFO dan WARNING
# Harus diletakkan sebelum import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64

# Meng-nonaktifkan warning matplotlib di environment non-GUI
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)

# --- MEMUAT MODEL (PASTIKAN PATH FILE BENAR) ---
try:
    # Asumsi model-model ini dilatih dengan 10 fitur
    model_daily = load_model('models/lstm_daily.keras')
    model_weekly = load_model('models/lstm_weekly.keras')
    model_monthly = load_model('models/lstm_monthly.keras')
except IOError as e:
    print(f"Error loading models: {e}")
    print("Pastikan Anda sudah melatih dan menyimpan model di direktori 'models/'")
    exit()

# --- FUNGSI-FUNGSI PEMBANTU ---

def prepare_data(df):
    """
    PERBAIKAN BESAR:
    Mempersiapkan data dengan 10 fitur, sama seperti saat training.
    """
    # 1. Pastikan nama kolom harga benar
    if 'Terakhir' in df.columns:
        price_col = 'Terakhir'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError("File CSV harus memiliki kolom 'Close' atau 'Terakhir'")

    # Menyesuaikan nama kolom lain jika ada perbedaan
    open_col = 'Pembukaan' if 'Pembukaan' in df.columns else 'Open'
    high_col = 'Tertinggi' if 'Tertinggi' in df.columns else 'High'
    low_col = 'Terendah' if 'Terendah' in df.columns else 'Low'

    # 2. Lakukan Feature Engineering yang sama persis seperti saat training
    df['Change_Open_Close'] = df[open_col] - df[price_col]
    df['Change_High_Low'] = df[high_col] - df[low_col]
    df['RollingMean_5'] = df[price_col].rolling(window=5).mean()
    df['RollingStd_5'] = df[price_col].rolling(window=5).std()
    df['RollingMean_10'] = df[price_col].rolling(window=10).mean()
    df['RollingStd_10'] = df[price_col].rolling(window=10).std()
    df['Return_1'] = df[price_col].pct_change(1)
    df['Return_5'] = df[price_col].pct_change(5)
    df['Return_10'] = df[price_col].pct_change(10)
    
    # 3. Isi missing values (NaN) yang muncul dari rolling mean/std/return
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True) # Tambahan untuk mengisi jika ada NaN di awal

    # 4. Definisikan fitur dan target
    features = [
        'Change_Open_Close', 'Change_High_Low',
        'RollingMean_5', 'RollingStd_5',
        'RollingMean_10', 'RollingStd_10',
        'Return_1', 'Return_5', 'Return_10'
    ]
    # Penting: Tambahkan harga itu sendiri sebagai fitur
    features_with_target = [price_col] + features 
    target_col = price_col

    # 5. Scaling dan pembuatan sequence
    data = df[features_with_target].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    window_size = 60 # Samakan dengan window_size saat training
    
    if len(data_scaled) <= window_size:
        return None, None, None

    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, :]) # Ambil semua fitur
        y.append(data_scaled[i, 0]) # Target adalah kolom pertama (harga)
        
    X, y = np.array(X), np.array(y)
    
    # Shape X sekarang akan menjadi (n_samples, window_size, 10), sesuai dengan yang diharapkan model
    return X, y, scaler

def evaluate_model(model, X, y, scaler):
    """Mengevaluasi model dan mengembalikan metrik serta nilai prediksi."""
    if X is None or len(X) == 0:
        return 0, 0, 0, 0, np.array([]), np.array([])
        
    prediction_scaled = model.predict(X)
    
    # Untuk inverse transform, kita butuh data dummy dengan shape (n_samples, n_features)
    # n_features adalah 10 dalam kasus ini
    n_features = X.shape[2] 
    
    # Inverse transform untuk prediksi
    dummy_pred = np.zeros((len(prediction_scaled), n_features))
    dummy_pred[:, 0] = prediction_scaled[:, 0]
    prediction_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]

    # Inverse transform untuk y_test
    dummy_y = np.zeros((len(y), n_features))
    dummy_y[:, 0] = y
    y_rescaled = scaler.inverse_transform(dummy_y)[:, 0]
    
    mae = np.mean(np.abs(y_rescaled - prediction_rescaled))
    mape = np.mean(np.abs((y_rescaled - prediction_rescaled) / y_rescaled)) * 100
    mse = np.mean((y_rescaled - prediction_rescaled)**2)
    rmse = np.sqrt(mse)
    
    return mae, mape, mse, rmse, y_rescaled, prediction_rescaled

def plot_to_base64(actual, predicted, title):
    """Membuat plot perbandingan dan mengonversinya ke gambar base64."""
    if len(actual) == 0 or len(predicted) == 0:
        return ""
        
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Harga Aktual', color='blue', alpha=0.7)
    plt.plot(predicted, label='Harga Prediksi', color='red', linestyle='--')
    plt.title(f'Perbandingan Harga Aktual vs Prediksi ({title})', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Harga', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

# --- ROUTE FLASK ---
# (Fungsi index() dan sisanya tetap sama, tidak perlu diubah)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        daily_file = request.files.get('daily')
        weekly_file = request.files.get('weekly')
        monthly_file = request.files.get('monthly')

        if not (daily_file and weekly_file and monthly_file):
            return "Harap unggah ketiga file (harian, mingguan, bulanan).", 400

        daily_df = pd.read_csv(daily_file)
        weekly_df = pd.read_csv(weekly_file)
        monthly_df = pd.read_csv(monthly_file)

        X_daily, y_daily, scaler_daily = prepare_data(daily_df)
        X_weekly, y_weekly, scaler_weekly = prepare_data(weekly_df)
        X_monthly, y_monthly, scaler_monthly = prepare_data(monthly_df)

        mae_d, mape_d, mse_d, rmse_d, y_true_daily, y_pred_daily = evaluate_model(model_daily, X_daily, y_daily, scaler_daily)
        mae_w, mape_w, mse_w, rmse_w, y_true_weekly, y_pred_weekly = evaluate_model(model_weekly, X_weekly, y_weekly, scaler_weekly)
        mae_m, mape_m, mse_m, rmse_m, y_true_monthly, y_pred_monthly = evaluate_model(model_monthly, X_monthly, y_monthly, scaler_monthly)
        
        plot_daily = plot_to_base64(y_true_daily, y_pred_daily, "Harian")
        plot_weekly = plot_to_base64(y_true_weekly, y_pred_weekly, "Mingguan")
        plot_monthly = plot_to_base64(y_true_monthly, y_pred_monthly, "Bulanan")

        results = [
            {"timeframe": "Harian", "mae": f'{mae_d:,.2f}', "mape": f'{mape_d:.2f}', "mse": f'{mse_d:,.2f}', "rmse": f'{rmse_d:,.2f}'},
            {"timeframe": "Mingguan", "mae": f'{mae_w:,.2f}', "mape": f'{mape_w:.2f}', "mse": f'{mse_w:,.2f}', "rmse": f'{rmse_w:,.2f}'},
            {"timeframe": "Bulanan", "mae": f'{mae_m:,.2f}', "mape": f'{mape_m:.2f}', "mse": f'{mse_m:,.2f}', "rmse": f'{rmse_m:,.2f}'}
        ]

        mape_values = {'Harian': mape_d, 'Mingguan': mape_w, 'Bulanan': mape_m}
        best_timeframe = min(mape_values, key=mape_values.get) if all(v > 0 for v in mape_values.values()) else "Tidak dapat ditentukan"

        return render_template('result.html',
                               results=results,
                               plot_daily=plot_daily,
                               plot_weekly=plot_weekly,
                               plot_monthly=plot_monthly,
                               best_timeframe=best_timeframe)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

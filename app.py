import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay, DateOffset

# =================================================================================
# Konfigurasi Halaman & Judul
# =================================================================================
st.set_page_config(page_title="Analisis Saham BBCA", layout="wide")
st.title("📈 Dasbor Analisis & Prediksi Saham BBCA")
st.markdown("Aplikasi ini menggunakan model LSTM untuk menganalisis data historis dalam tiga timeframe.")

# =================================================================================
# CACHING: Fungsi untuk memuat data dan model
# =================================================================================

# Cache untuk memuat model Keras
@st.cache_resource
def load_keras_model(timeframe):
    """Memuat model Keras dari file lokal berdasarkan timeframe."""
    path = f'models/lstm_{timeframe}.keras'
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error memuat model dari path: {path}. Pastikan file ada. Error: {e}")
        return None

# Cache untuk memuat dan memproses data CSV
@st.cache_data
def load_and_process_data(timeframe):
    """Memuat data CSV dari file lokal dan memprosesnya."""
    path = f'data/Data Historis BBCA {timeframe}.csv'
    try:
        df = pd.read_csv(path, decimal=',', thousands='.')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)
        df.sort_values('Tanggal', inplace=True)
        df.set_index('Tanggal', inplace=True)

        # Feature Engineering
        df['Change_Open_Close'] = df['Pembukaan'] - df['Terakhir']
        df['Change_High_Low'] = df['Tertinggi'] - df['Terendah']
        df['RollingMean_5'] = df['Terakhir'].rolling(window=5).mean()
        df['RollingStd_5'] = df['Terakhir'].rolling(window=5).std()
        df['RollingMean_10'] = df['Terakhir'].rolling(window=10).mean()
        df['RollingStd_10'] = df['Terakhir'].rolling(window=10).std()
        df['Return_1'] = df['Terakhir'].pct_change(1)
        df['Return_5'] = df['Terakhir'].pct_change(5)
        df['Return_10'] = df['Terakhir'].pct_change(10)
        
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error memuat data dari path: {path}. Pastikan file CSV ada. Error: {e}")
        return None

# =================================================================================
# Fungsi Logika Prediksi
# =================================================================================

def predict_future(model, df, feature_cols, target_col, scaler, window_size, n_steps):
    """Fungsi untuk melakukan prediksi masa depan."""
    last_window_data = df[feature_cols].iloc[-window_size:].values
    future_predictions = []

    for _ in range(n_steps):
        scaled_window = scaler.transform(last_window_data)
        X_to_predict = scaled_window.reshape(1, window_size, len(feature_cols))
        predicted_scaled = model.predict(X_to_predict, verbose=0)[0, 0]

        dummy_for_inverse = np.zeros((1, len(feature_cols)))
        target_idx = feature_cols.index(target_col)
        dummy_for_inverse[0, target_idx] = predicted_scaled
        predicted_price = scaler.inverse_transform(dummy_for_inverse)[0, target_idx]
        future_predictions.append(predicted_price)

        new_row_features = last_window_data[-1, :].copy()
        new_row_features[target_idx] = predicted_price
        last_window_data = np.vstack([last_window_data[1:], new_row_features])
        
    return future_predictions

# =================================================================================
# Sidebar - Panel Kontrol
# =================================================================================
st.sidebar.header("⚙️ Panel Kontrol")

timeframe_options = {'Harian': 'daily', 'Mingguan': 'weekly', 'Bulanan': 'monthly'}
timeframe_selection = st.sidebar.radio(
    "Pilih Timeframe Analisis:",
    options=list(timeframe_options.keys()),
)
timeframe_code = timeframe_options[timeframe_selection]

# Atur parameter berdasarkan timeframe
if timeframe_code == 'daily':
    n_future = st.sidebar.slider("Jumlah Hari Prediksi:", 5, 30, 10, key='d')
    window_size = 60
elif timeframe_code == 'weekly':
    n_future = st.sidebar.slider("Jumlah Minggu Prediksi:", 4, 24, 8, key='w')
    window_size = 20
else: # monthly
    n_future = st.sidebar.slider("Jumlah Bulan Prediksi:", 3, 18, 12, key='m')
    window_size = 12

run_button = st.sidebar.button("🚀 Jalankan Analisis")

# =================================================================================
# Area Utama - Tampilan Hasil
# =================================================================================


if run_button:
    with st.spinner(f"Memproses analisis untuk timeframe **{timeframe_selection}**..."):
        # 1. Muat model dan data
        model = load_keras_model(timeframe_code)
        df = load_and_process_data(timeframe_code)
        
        if model is not None and df is not None:
            
            # =======================================================
            # === TAMBAHKAN BLOK PENGECEKAN INI ===
            # =======================================================
            if len(df) < window_size:
                st.error(f"Error: Data tidak cukup untuk prediksi.")
                st.error(f"Timeframe '{timeframe_selection}' membutuhkan minimal {window_size} baris data, tetapi setelah diproses hanya tersedia {len(df)} baris.")
                st.info("Silakan periksa file CSV sumber Anda atau pilih timeframe yang berbeda.")
            
            # =======================================================
            # === LANJUTKAN HANYA JIKA DATA CUKUP ===
            # =======================================================
            else:
                # 2. Siapkan data untuk model
                target_col = 'Terakhir'
                feature_cols = ['Terakhir', 'Change_Open_Close', 'Change_High_Low', 'RollingMean_5', 'RollingStd_5', 'RollingMean_10', 'RollingStd_10', 'Return_1', 'Return_5', 'Return_10']
                
                scaler = MinMaxScaler()
                scaler.fit(df[feature_cols])

                # 3. Lakukan Prediksi
                future_preds = predict_future(model, df, feature_cols, target_col, scaler, window_size, n_future)
                
            # 4. Siapkan data untuk tabel dan ringkasan
            last_real_date = df.index[-1]
            if timeframe_code == 'daily':
                future_dates = pd.to_datetime([last_real_date + BDay(i) for i in range(1, n_future + 1)])
            elif timeframe_code == 'weekly':
                future_dates = pd.to_datetime([last_real_date + DateOffset(weeks=i) for i in range(1, n_future + 1)])
            else: # monthly
                future_dates = pd.to_datetime([last_real_date + DateOffset(months=i) for i in range(1, n_future + 1)])

            # 5. Buat Ringkasan Utama
            last_actual_price = df['Terakhir'].iloc[-1]
            first_predicted_price = future_preds[0]
            change_pct = (first_predicted_price - last_actual_price) / last_actual_price
            rekomendasi_utama = "HOLD 😐"
            if change_pct > 0.005: rekomendasi_utama = "BUY 🟢"
            elif change_pct < -0.005: rekomendasi_utama = "SELL 🔴"

            st.subheader("Ringkasan Prediksi")
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir (dari CSV)", f"Rp {last_actual_price:,.0f}")
            col2.metric("Prediksi Berikutnya", f"Rp {first_predicted_price:,.0f}", f"{change_pct:.2%}")
            col3.metric("Rekomendasi Utama", rekomendasi_utama)
            st.divider()

            # 6. Buat TABS
            tab1, tab2 = st.tabs(["📊 Grafik Prediksi", "📋 Tabel Proyeksi"])

            with tab1:
                st.subheader(f"Grafik Prediksi {n_future} {timeframe_selection} ke Depan")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df.index[-120:], df['Terakhir'][-120:], label='Harga Historis Aktual', color='dodgerblue')
                ax.plot(future_dates, future_preds, label=f'Prediksi', color='orange', marker='o', linestyle='--')
                ax.set_title(f'Prediksi Harga Saham BBCA - {timeframe_selection}')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Harga (Rp)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            with tab2:
                st.subheader("Tabel Detail Proyeksi dan Rekomendasi")
                prediksi_df = pd.DataFrame({'Tanggal': future_dates, 'Harga Prediksi': future_preds})
                prediksi_df['Tanggal'] = prediksi_df['Tanggal'].dt.strftime('%d %B %Y')
                prediksi_df['Harga Prediksi (Rp)'] = prediksi_df['Harga Prediksi'].apply(lambda x: f"{x:,.0f}")
                
                rekomendasi_tabel = []
                current_price = last_actual_price
                for pred in future_preds:
                    change = (pred - current_price) / current_price
                    if change > 0.005: rekomendasi_tabel.append('BUY 🟢')
                    elif change < -0.005: rekomendasi_tabel.append('SELL 🔴')
                    else: rekomendasi_tabel.append('HOLD 😐')
                    current_price = pred
                prediksi_df['Rekomendasi'] = rekomendasi_tabel
                st.dataframe(prediksi_df, use_container_width=True)
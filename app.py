import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIGURATION ---
# Setting the page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Analisis Saham BBCA",
    page_icon="📈",
    layout="wide"
)

# --- HELPER FUNCTIONS (SAME AS FLASK, BUT WITH CACHE DECORATORS) ---

# @st.cache_resource decorator is used to load the models only once, saving memory and time.
@st.cache_resource
def load_all_models():
    """Loads all .keras models from the /models folder."""
    # Handle path issues in Streamlit Cloud
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, 'models')
    
    try:
        model_daily = load_model(os.path.join(models_path, 'lstm_daily.keras'))
        model_weekly = load_model(os.path.join(models_path, 'lstm_weekly.keras'))
        model_monthly = load_model(os.path.join(models_path, 'lstm_monthly.keras'))
        return model_daily, model_weekly, model_monthly
    except Exception as e:
        st.error(f"Gagal memuat model: {e}. Pastikan file model ada di dalam folder 'models'.")
        return None, None, None

def prepare_data(df):
    """Prepares data with 10 features, same as during training."""
    if 'Terakhir' in df.columns:
        price_col = 'Terakhir'
    elif 'Close' in df.columns:
        price_col = 'Close'
    else:
        raise ValueError("File CSV harus memiliki kolom 'Close' atau 'Terakhir'")

    open_col = 'Pembukaan' if 'Pembukaan' in df.columns else 'Open'
    high_col = 'Tertinggi' if 'Tertinggi' in df.columns else 'High'
    low_col = 'Terendah' if 'Terendah' in df.columns else 'Low'

    numeric_cols = [price_col, open_col, high_col, low_col]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Change_Open_Close'] = df[open_col] - df[price_col]
    df['Change_High_Low'] = df[high_col] - df[low_col]
    df['RollingMean_5'] = df[price_col].rolling(window=5).mean()
    df['RollingStd_5'] = df[price_col].rolling(window=5).std()
    df['RollingMean_10'] = df[price_col].rolling(window=10).mean()
    df['RollingStd_10'] = df[price_col].rolling(window=10).std()
    df['Return_1'] = df[price_col].pct_change(1)
    df['Return_5'] = df[price_col].pct_change(5)
    df['Return_10'] = df[price_col].pct_change(10)

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    features = [
        'Change_Open_Close', 'Change_High_Low', 'RollingMean_5', 'RollingStd_5',
        'RollingMean_10', 'RollingStd_10', 'Return_1', 'Return_5', 'Return_10'
    ]
    features_with_target = [price_col] + features
    
    data = df[features_with_target].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    window_size = 60
    if len(data_scaled) <= window_size:
        return None, None, None

    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, :])
        y.append(data_scaled[i, 0])

    return np.array(X), np.array(y), scaler

def evaluate_model(model, X, y, scaler):
    """Evaluates the model and returns metrics and prediction values."""
    if X is None or len(X) == 0:
        return 0, 0, 0, 0, np.array([]), np.array([])
    
    prediction_scaled = model.predict(X)
    n_features = X.shape[2]

    dummy_pred = np.zeros((len(prediction_scaled), n_features))
    dummy_pred[:, 0] = prediction_scaled[:, 0]
    prediction_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_y = np.zeros((len(y), n_features))
    dummy_y[:, 0] = y
    y_rescaled = scaler.inverse_transform(dummy_y)[:, 0]
    
    mae = np.mean(np.abs(y_rescaled - prediction_rescaled))
    mape = np.mean(np.abs((y_rescaled - prediction_rescaled) / y_rescaled)) * 100
    mse = np.mean((y_rescaled - prediction_rescaled)**2)
    rmse = np.sqrt(mse)
    
    return mae, mape, mse, rmse, y_rescaled, prediction_rescaled

def create_plot(actual, predicted, title):
    """Creates and returns a Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label='Harga Aktual', color='blue', alpha=0.7)
    ax.plot(predicted, label='Harga Prediksi', color='red', linestyle='--')
    ax.set_title(f'Perbandingan Harga Aktual vs Prediksi ({title})', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Harga', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

# --- STREAMLIT INTERFACE ---

# Main Title and Description
st.title("📈 Analisis Prediksi Saham BBCA dengan LSTM")
st.write(
    "Aplikasi ini menggunakan model Long Short-Term Memory (LSTM) untuk memprediksi harga saham BBCA. "
    "Silakan unggah tiga file data historis (.csv) untuk timeframe Harian, Mingguan, dan Bulanan untuk memulai."
)

# Load models once and cache them
model_daily, model_weekly, model_monthly = load_all_models()

# Initialize session state to store results
if 'results' not in st.session_state:
    st.session_state['results'] = None

# File Upload Section in the Sidebar
with st.sidebar:
    
    st.markdown(
        unsafe_allow_html=True,
    )
    
    try:
        # Membuat path yang benar ke gambar, agar berfungsi baik di lokal maupun saat deploy
        base_path = os.path.dirname(__file__)
        image_path = os.path.join(base_path, 'static', 'meme-stonks.jpg')
        st.image(image_path, width=150)
    except FileNotFoundError:
        st.warning("File 'meme-stonks.jpg' tidak ditemukan. Pastikan file tersebut ada di dalam folder 'static'.")

    daily_file = st.file_uploader("Data Harian (.csv)", type="csv")
    weekly_file = st.file_uploader("Data Mingguan (.csv)", type="csv")
    monthly_file = st.file_uploader("Data Bulanan (.csv)", type="csv")

    analyze_button = st.button("Analisis Sekarang", type="primary", use_container_width=True)

# Logic when the analyze button is pressed
if analyze_button:
    if daily_file and weekly_file and monthly_file and model_daily:
        with st.spinner('Sedang menganalisis data dan membuat prediksi... Ini mungkin memakan waktu beberapa saat.'):
            try:
                # Read data
                daily_df = pd.read_csv(daily_file)
                weekly_df = pd.read_csv(weekly_file)
                monthly_df = pd.read_csv(monthly_file)

                # Process and evaluate each timeframe
                X_d, y_d, s_d = prepare_data(daily_df)
                mae_d, mape_d, mse_d, rmse_d, y_true_d, y_pred_d = evaluate_model(model_daily, X_d, y_d, s_d)

                X_w, y_w, s_w = prepare_data(weekly_df)
                mae_w, mape_w, mse_w, rmse_w, y_true_w, y_pred_w = evaluate_model(model_weekly, X_w, y_w, s_w)

                X_m, y_m, s_m = prepare_data(monthly_df)
                mae_m, mape_m, mse_m, rmse_m, y_true_m, y_pred_m = evaluate_model(model_monthly, X_m, y_m, s_m)

                # Determine the best timeframe
                mape_values = {'Harian': mape_d, 'Mingguan': mape_w, 'Bulanan': mape_m}
                best_timeframe = min(mape_values, key=mape_values.get) if all(v > 0 for v in mape_values.values()) else "N/A"

                # Save all results to session state
                st.session_state['results'] = {
                    "daily": {"mae": mae_d, "mape": mape_d, "mse": mse_d, "rmse": rmse_d, "true": y_true_d, "pred": y_pred_d},
                    "weekly": {"mae": mae_w, "mape": mape_w, "mse": mse_w, "rmse": rmse_w, "true": y_true_w, "pred": y_pred_w},
                    "monthly": {"mae": mae_m, "mape": mape_m, "mse": mse_m, "rmse": rmse_m, "true": y_true_m, "pred": y_pred_m},
                    "best": best_timeframe
                }
                st.success("Analisis selesai!")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan data: {e}")
                st.session_state['results'] = None
    else:
        st.warning("Harap unggah ketiga file CSV sebelum menekan tombol analisis.")

# Display results if they exist in the session state
if st.session_state['results']:
    results = st.session_state['results']
    best_timeframe = results['best']
    
    st.header("Hasil Evaluasi")

    # Display the best timeframe
    st.subheader("Timeframe Prediksi Terbaik")
    st.metric(label="Timeframe Terbaik (MAPE Terendah)", value=best_timeframe)
    st.markdown("---")

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Harian")
        st.metric("MAPE", f"{results['daily']['mape']:.2f}%")
        st.metric("RMSE", f"{results['daily']['rmse']:,.2f}")
        st.metric("MAE", f"{results['daily']['mae']:,.2f}")
        st.metric("MSE", f"{results['daily']['mse']:,.2f}")
    with col2:
        st.subheader("Mingguan")
        st.metric("MAPE", f"{results['weekly']['mape']:.2f}%")
        st.metric("RMSE", f"{results['weekly']['rmse']:,.2f}")
        st.metric("MAE", f"{results['weekly']['mae']:,.2f}")
        st.metric("MSE", f"{results['weekly']['mse']:,.2f}")
    with col3:
        st.subheader("Bulanan")
        st.metric("MAPE", f"{results['monthly']['mape']:.2f}%")
        st.metric("RMSE", f"{results['monthly']['rmse']:,.2f}")
        st.metric("MAE", f"{results['monthly']['mae']:,.2f}")
        st.metric("MSE", f"{results['monthly']['mse']:,.2f}")
    st.markdown("---")
    
    # Display plots in tabs
    st.header("Grafik Prediksi vs. Aktual")
    tab1, tab2, tab3 = st.tabs(["Harian", "Mingguan", "Bulanan"])
    with tab1:
        fig_d = create_plot(results['daily']['true'], results['daily']['pred'], "Harian")
        st.pyplot(fig_d)
    with tab2:
        fig_w = create_plot(results['weekly']['true'], results['weekly']['pred'], "Mingguan")
        st.pyplot(fig_w)
    with tab3:
        fig_m = create_plot(results['monthly']['true'], results['monthly']['pred'], "Bulanan")
        st.pyplot(fig_m)

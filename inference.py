import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def run_predictions():
    print("1. Đang tải dữ liệu môi trường mới nhất...")
    df = pd.read_csv("processed_aqi_data.csv")
    
    # Lấy dòng dữ liệu gần nhất để dự báo
    latest_data = df.iloc[[-1]].copy()
    features = latest_data.drop(columns=['time', 'AQI', 'AQI_RF_Predict', 'AQI_LSTM_Predict'], errors='ignore')
    
    # --- Dự báo bằng Random Forest ---
    try:
        rf_model = joblib.load('rf_model.pkl') # Yêu cầu Vai 2 xuất file này
        df.loc[df.index[-1], 'AQI_RF_Predict'] = rf_model.predict(features)[0]
        print("✅ Đã dự báo xong Random Forest")
    except FileNotFoundError:
        print("⚠️ Chưa có file rf_model.pkl")

    # --- Dự báo bằng LSTM ---
    try:
        lstm_model = load_model('lstm_aqi_model.h5')
        # LSTM cần Tensor 3D: [1 mẫu, 1 timestep, số đặc trưng]
        lstm_features = np.array([features.values])
        df.loc[df.index[-1], 'AQI_LSTM_Predict'] = lstm_model.predict(lstm_features)[0][0]
        print("✅ Đã dự báo xong LSTM")
    except FileNotFoundError:
        print("⚠️ Chưa có file lstm_aqi_model.h5")
        
    # Ghi đè lại vào file CSV
    df.to_csv("processed_aqi_data.csv", index=False)
    print("💾 Đã cập nhật kết quả dự báo vào cơ sở dữ liệu!")

if __name__ == "__main__":
    run_predictions()
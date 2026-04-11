import pandas as pd
import numpy as np
import joblib
import os
# Lệnh này giúp tắt bớt các cảnh báo xám vô hại của TensorFlow cho Terminal gọn gàng
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

def run_predictions():
    print("1. Đang tải dữ liệu môi trường mới nhất...")
    df = pd.read_csv("processed_aqi_data.csv")
    
    # Kịch bản bảo vệ: Nếu kho dữ liệu chưa gom đủ 4 giờ, mô hình không thể chạy sliding window
    if len(df) < 4:
        print("⚠️ Chưa đủ dữ liệu quá khứ để tạo cửa sổ trượt (cần ít nhất 4 dòng). Sẽ bỏ qua lần dự báo này.")
        return

    # ---------------------------------------------------------
    # CHUẨN BỊ DỮ LIỆU CHO RANDOM FOREST (Cần 3 cột Lag Features)
    # ---------------------------------------------------------
    # Trích xuất 4 giờ gần nhất để tính phép dịch chuyển (shift)
    recent_4_rows = df.tail(4).copy()
    for i in range(1, 4):
        recent_4_rows[f'AQI_lag_{i}'] = recent_4_rows['AQI'].shift(i)
    
    # Lấy dòng cuối cùng (tức là giờ hiện tại) sau khi đã ghép đủ 3 cột quá khứ
    rf_features = recent_4_rows.iloc[[-1]].drop(columns=['time', 'AQI', 'AQI_RF_Predict', 'AQI_LSTM_Predict'], errors='ignore')

    try:
        rf_model = joblib.load('rf_model.pkl')
        # Ép thứ tự các cột phải giống 100% như lúc huấn luyện
        expected_features = rf_model.feature_names_in_ 
        rf_input = rf_features[expected_features]
        
        df.loc[df.index[-1], 'AQI_RF_Predict'] = rf_model.predict(rf_input)[0]
        print("✅ Đã dự báo xong Random Forest")
    except FileNotFoundError:
        print("⚠️ Chưa có file rf_model.pkl")

    # ---------------------------------------------------------
    # CHUẨN BỊ DỮ LIỆU CHO LSTM (Cần Tensor 3 chiều)
    # ---------------------------------------------------------
    # Lấy đúng 3 dòng gần nhất (không lấy dòng hiện tại) để dự báo tương lai
    lstm_feature_df = df.tail(3).drop(columns=['time', 'AQI', 'AQI_RF_Predict', 'AQI_LSTM_Predict'], errors='ignore')

    try:
        lstm_model = load_model('lstm_aqi_model.h5')
        # Biến bảng tính (2D) thành không gian Tensor (3D)
        lstm_input = np.array([lstm_feature_df.values]) 
        df.loc[df.index[-1], 'AQI_LSTM_Predict'] = lstm_model.predict(lstm_input, verbose=0)[0][0]
        print("✅ Đã dự báo xong LSTM")
    except FileNotFoundError:
        print("⚠️ Chưa có file lstm_aqi_model.h5")
        
    # Ghi đè lại kết quả vào cơ sở dữ liệu
    df.to_csv("processed_aqi_data.csv", index=False)
    print("💾 Đã cập nhật kết quả dự báo vào cơ sở dữ liệu!")

if __name__ == "__main__":
    run_predictions()
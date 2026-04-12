import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    print("1. Đang tải dữ liệu thô...")
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    
    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 1: NỘI SUY (INTERPOLATION) LẤP ĐẦY DỮ LIỆU THIẾU
    # ---------------------------------------------------------
    # Sử dụng nội suy tuyến tính để điền các giá trị NaN (nếu có do lỗi API hoặc cảm biến)
    print("2. Đang xử lý dữ liệu khuyết thiếu...")
    df.interpolate(method='linear', inplace=True)
    
    # Điền các giá trị NaN ở đầu hoặc cuối bảng (nội suy không tới được)
    df.bfill(inplace=True) 
    df.ffill(inplace=True)

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 2: TÍNH CHỈ SỐ AQI TỪ NỒNG ĐỘ PM2.5
    # ---------------------------------------------------------
    print("3. Đang tính toán chỉ số AQI...")
    def calculate_aqi(pm25):
        # Hàm phân đoạn tuyến tính theo tiêu chuẩn EPA Hoa Kỳ
        breakpoints = [
            (0.0, 12.0, 0, 50),         # Tốt
            (12.1, 35.4, 51, 100),      # Trung bình
            (35.5, 55.4, 101, 150),     # Kém
            (55.5, 150.4, 151, 200),    # Xấu
            (150.5, 250.4, 201, 300),   # Rất xấu
            (250.5, 350.4, 301, 400),   # Nguy hại
            (350.5, 500.4, 401, 500)    # Nguy hại
        ]
        for (c_low, c_high, i_low, i_high) in breakpoints:
            if c_low <= pm25 <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
        return 500 # Vượt ngưỡng
        
    df['AQI'] = df['pm2_5'].apply(calculate_aqi).round(0)

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 3: KỸ THUẬT ĐẶC TRƯNG THỜI GIAN (CYCLICAL FEATURES)
    # ---------------------------------------------------------
    print("4. Đang trích xuất đặc trưng chu kỳ thời gian...")
    df['hour'] = df['time'].dt.hour
    
    # Biến đổi lượng giác (Sine/Cosine) để mô hình hiểu giờ 23 và giờ 0 là liền kề nhau
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23.0)

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 4: CHUẨN HÓA DỮ LIỆU (STANDARDIZATION)
    # ---------------------------------------------------------
    print("5. Đang chuẩn hóa các biến số môi trường...")
    # Chọn các cột đặc trưng cần chuẩn hóa (không chuẩn hóa AQI vì nó là biến mục tiêu)
    features_to_scale = ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone', 'carbon_monoxide']
    
    scaler = StandardScaler()
    # Định tâm các biến về trung bình 0 và phương sai 1
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df

if __name__ == "__main__":
    try:
        # Chạy hàm xử lý để có dữ liệu sạch mới nhất
        processed_df = load_and_clean_data("raw_aqi_data.csv")
        
        # ========================================================
        # ĐOẠN CODE CỨU HỘ: BẢO TOÀN LỊCH SỬ DỰ BÁO CỦA AI (UPSERT)
        # ========================================================
        try:
            print("6. Đang kiểm tra và bảo tồn lịch sử dự báo cũ...")
            # Đọc lại file processed cũ để "cứu" lịch sử
            old_processed = pd.read_csv('processed_aqi_data.csv')
            
            # Chỉ trích xuất cột thời gian và 2 cột dự báo
            old_predictions = old_processed[['time', 'AQI_RF_Predict', 'AQI_LSTM_Predict']]
            
            # Đảm bảo 2 cột time cùng định dạng datetime trước khi nối để không bị lệch
            processed_df['time'] = pd.to_datetime(processed_df['time'])
            old_predictions['time'] = pd.to_datetime(old_predictions['time'])
            
            # Nối (Left Join) các cột dự báo cũ sang bảng dữ liệu mới
            processed_df = pd.merge(processed_df, old_predictions, on='time', how='left')
            print("🔄 Đã ghép nối thành công lịch sử dự báo cũ!")
        except (FileNotFoundError, KeyError):
            # Nếu chạy lần đầu chưa có file hoặc chưa có cột dự báo thì bỏ qua
            print("🆕 Chạy lần đầu hoặc chưa có lịch sử dự báo. Đang tạo mới...")
        # ========================================================
        
        # Lưu ra file mới (lúc này đã có cả data mới và lịch sử dự báo cũ)
        processed_df.to_csv("processed_aqi_data.csv", index=False)
        
        print("\n✅ Hoàn tất! Xem thử 5 dòng dữ liệu MỚI NHẤT (Tail):")
        # In tail() thay vì head() để thấy rõ dòng hiện tại các cột Predict đang bị trống (NaN) chờ AI điền vào
        print(processed_df[['time', 'AQI', 'AQI_RF_Predict', 'AQI_LSTM_Predict']].tail())
        print("\n💾 Đã lưu thành công ra file 'processed_aqi_data.csv'")
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file 'raw_aqi_data.csv'. Vui lòng đảm bảo Kịch bản 1 đã chạy thành công.")
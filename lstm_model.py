import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# TRỌNG ĐIỂM 1: CẤU TRÚC LẠI DỮ LIỆU THÀNH TENSOR 3 CHIỀU
# ---------------------------------------------------------
def create_sequences(features, target, seq_length):
    """
    Biến đổi dữ liệu dạng bảng (2D) thành Tensor 3 chiều: [samples, timesteps, features]
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features.iloc[i:(i + seq_length)].values)
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)

def train_and_evaluate_lstm(filepath):
    print("1. Đang tải và chuẩn bị dữ liệu cho mô hình Deep Learning...")
    df = pd.read_csv(filepath)
    
    # Tách biến độc lập (X) và biến phụ thuộc/mục tiêu (y)
    features = df.drop(columns=['time', 'AQI'])
    target = df['AQI']
    
    # Thiết lập cửa sổ thời gian (Sliding Window): Nhìn lại 3 giờ trước
    SEQ_LENGTH = 3
    X, y = create_sequences(features, target, SEQ_LENGTH)
    
    print(f"   -> Đã định dạng thành Tensor 3 chiều với kích thước: {X.shape}")
    # X.shape sẽ có dạng (Số lượng mẫu, 3 bước thời gian, Số lượng đặc trưng)

    # Chia tập dữ liệu (Không trộn xáo ngẫu nhiên, giữ nguyên trục thời gian)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 2: THIẾT KẾ KIẾN TRÚC MẠNG NƠ-RON LSTM
    # ---------------------------------------------------------
    print("2. Đang xây dựng và biên dịch mạng LSTM...")
    model = Sequential()
    
    # Lớp LSTM đầu tiên với tham số return_sequences=True để truyền chuỗi sang lớp tiếp theo
    model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    # Lớp Dropout để ngẫu nhiên vô hiệu hóa 20% nơ-ron, chống lại hiện tượng quá khớp (Overfitting)
    model.add(Dropout(0.2))
    
    # Lớp LSTM thứ hai, trích xuất đặc trưng sâu hơn
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.2))
    
    # Lớp đầu ra (Dense Layer) tuyến tính để dự báo 1 giá trị AQI duy nhất
    model.add(Dense(units=1))

    # Biên dịch mô hình với thuật toán Adam và hàm mất mát Sai số Toàn phương Trung bình (MSE)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 3: HUẤN LUYỆN (TRAINING)
    # ---------------------------------------------------------
    print("3. Đang tiến hành huấn luyện (Training)...")
    # Sử dụng batch_size và epochs để kiểm soát tốc độ học
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 4: ĐÁNH GIÁ (EVALUATION)
    # ---------------------------------------------------------
    print("\n4. Đang đưa ra dự báo và đối chiếu độ chính xác...")
    y_pred = model.predict(X_test).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=========================================")
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LSTM")
    print("=========================================")
    print(f"RMSE (Căn bậc hai sai số toàn phương): {rmse:.2f}")
    print(f"MAE  (Sai số tuyệt đối trung bình):    {mae:.2f}")
    print(f"R²   (Hệ số xác định):                 {r2:.4f}")
    
    # Lưu mô hình lại để sau này Dashboard có thể tái sử dụng mà không cần train lại
    model.save('lstm_aqi_model.h5')
    print("\n💾 Đã lưu mạng nơ-ron thành công vào file 'lstm_aqi_model.h5'")

if __name__ == "__main__":
    try:
        train_and_evaluate_lstm("processed_aqi_data.csv")
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
        print("Mẹo: Mạng nơ-ron cần rất nhiều dữ liệu để học. Nếu báo lỗi thiếu mẫu, hãy đợi GitHub Actions gom thêm dữ liệu nhé!")
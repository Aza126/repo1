import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def create_lag_features(df, target_col='AQI', lags=3):
    """
    Áp dụng kỹ thuật Cửa sổ trượt (Sliding Window).
    Tạo ra các cột dữ liệu của t-1, t-2, t-3 để dự báo cho thời điểm t.
    """
    print(f"1. Đang tái cấu trúc dữ liệu với cửa sổ trượt {lags} giờ...")
    df_lagged = df.copy()
    for i in range(1, lags + 1):
        df_lagged[f'{target_col}_lag_{i}'] = df_lagged[target_col].shift(i)
    
    # Xóa bỏ các dòng bị NaN do phép toán dịch chuyển (shift) tạo ra
    df_lagged.dropna(inplace=True)
    return df_lagged

def train_and_evaluate_rf(filepath):
    # Đọc dữ liệu đã qua tiền xử lý
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    
    # Khởi tạo cửa sổ trượt (nhìn lại 3 giờ trước)
    df_model = create_lag_features(df, target_col='AQI', lags=3)
    
    # Xác định biến đầu vào (X) và biến mục tiêu (y)
    # Loại bỏ cột 'time' (vì thuật toán chỉ hiểu số) và cột mục tiêu 'AQI' khỏi X
    X = df_model.drop(columns=['time', 'AQI'])
    y = df_model['AQI']
    
    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 1: CHIA TẬP DỮ LIỆU THEO CHUỖI THỜI GIAN
    # ---------------------------------------------------------
    # Không được chia ngẫu nhiên, phải chia theo trình tự thời gian (80% quá khứ để train, 20% tương lai để test)
    split_index = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"2. Kích thước tập Huấn luyện: {len(X_train)} mẫu | Tập Kiểm thử: {len(X_test)} mẫu")

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 2: TÌM KIẾM LƯỚI & KIỂM ĐỊNH CHÉO THỜI GIAN
    # ---------------------------------------------------------
    print("3. Đang tinh chỉnh siêu tham số và huấn luyện Random Forest...")
    rf = RandomForestRegressor(random_state=42)
    
    # Cấu hình không gian tham số để tìm kiếm (Grid Search)
    param_grid = {
        'n_estimators': [50, 100],        # Số lượng cây quyết định trong rừng
        'max_depth': [None, 10, 20],      # Độ sâu tối đa của cây
        'min_samples_split': [2, 5]       # Số mẫu tối thiểu để phân nhánh
    }
    
    # Sử dụng TimeSeriesSplit thay vì K-Fold thông thường
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    print(f"   -> Siêu tham số tối ưu tìm được: {grid_search.best_params_}")

    joblib.dump(best_rf, 'rf_model.pkl')
    print("   -> Đã xuất mô hình ra file 'rf_model.pkl' thành công!")

    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 3: ĐÁNH GIÁ HIỆU NĂNG BẰNG CÁC THƯỚC ĐO CHUẨN
    # ---------------------------------------------------------
    y_pred = best_rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=========================================")
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST")
    print("=========================================")
    print(f"RMSE (Căn bậc hai sai số toàn phương): {rmse:.2f}")
    print(f"MAE  (Sai số tuyệt đối trung bình):    {mae:.2f}")
    print(f"R²   (Hệ số xác định):                 {r2:.4f}")
    
    # ---------------------------------------------------------
    # TRỌNG ĐIỂM 4: TRÍCH XUẤT ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG
    # ---------------------------------------------------------
    feature_importance = pd.DataFrame({
        'Đặc trưng': X.columns,
        'Mức độ đóng góp (%)': best_rf.feature_importances_ * 100
    }).sort_values(by='Mức độ đóng góp (%)', ascending=False)
    
    print("\nTOP 5 YẾU TỐ ẢNH HƯỞNG ĐẾN AQI NHẤT:")
    print(feature_importance.head(5).to_string(index=False))

if __name__ == "__main__":
    try:
        train_and_evaluate_rf("processed_aqi_data.csv")
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file processed_aqi_data.csv.")
    except ValueError as e:
        print(f"❌ Lỗi dữ liệu quá ít: {e}")
        print("Mẹo: Hãy để Kịch bản 1 và 2 chạy gom dữ liệu thêm khoảng 1-2 ngày nữa để có đủ data huấn luyện nhé!")
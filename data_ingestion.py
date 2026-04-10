import requests
import pandas as pd
from datetime import datetime

def fetch_environmental_data(lat, lon):
    # Đường dẫn URL của Open-Meteo API
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Thiết lập các tham số (vĩ độ, kinh độ, và các biến số mục tiêu)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "carbon_monoxide"],
        "timezone": "Asia/Bangkok",
        "past_days": 1 # Lấy thêm dữ liệu 1 ngày trước đó
    }
    
    try:
        # Gửi truy vấn GET đến máy chủ 
        print("Đang kết nối đến Open-Meteo API...")
        response = requests.get(url, params=params)
        
        # Bắt lỗi nếu mất kết nối mạng hoặc sai URL
        response.raise_for_status() 
        
        # Trích xuất dữ liệu JSON
        data = response.json()
        
        # "Làm phẳng" (flatten) cấu trúc JSON thành DataFrame đa chiều
        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)
        
        # Chuyển đổi cột 'time' sang định dạng Datetime của Pandas
        df['time'] = pd.to_datetime(df['time'])
        
        print(f"✅ Đã tải thành công {len(df)} dòng dữ liệu!")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi kết nối API: {e}")
        return None

# Kịch bản chạy thử nghiệm
if __name__ == "__main__":
    # Tọa độ tham khảo tại Hà Đông, Hà Nội
    LAT = 20.9716  
    LON = 105.7725 
    
    df_aqi = fetch_environmental_data(LAT, LON)
    
    if df_aqi is not None:
        # In ra 5 dòng đầu tiên để kiểm tra
        print("\nTrích xuất dữ liệu thành công:")
        print(df_aqi.head())
        
        # Lưu trữ cục bộ để phục vụ cho bước làm sạch dữ liệu sau này
        df_aqi.to_csv("raw_aqi_data.csv", index=False)
        print("\n💾 Đã lưu dữ liệu thô ra file 'raw_aqi_data.csv'")
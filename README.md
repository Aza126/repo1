# 🌍 Hệ thống Giám sát & Dự báo Chất lượng Không khí (AQI) Thời gian thực

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)

## 📖 Giới thiệu Dự án
Dự án này là một hệ thống dữ liệu khép kín (End-to-End Data Pipeline) được xây dựng để thu thập, xử lý, dự báo và trực quan hóa Chỉ số Chất lượng Không khí (AQI) theo thời gian thực. 

Thay vì sử dụng các tập dữ liệu tĩnh (file CSV tải sẵn), hệ thống được tự động hóa hoàn toàn bằng **GitHub Actions**, hoạt động bền bỉ 24/7 để cập nhật số liệu mới nhất mỗi giờ, đồng thời sử dụng Trí tuệ Nhân tạo để dự báo xu hướng ô nhiễm trong tương lai.

## ✨ Tính năng Nổi bật
- **Luồng dữ liệu tự động (Automated ETL):** Thu thập dữ liệu thời tiết và PM2.5 từ Open-Meteo API mỗi giờ.
- **Tiền xử lý thông minh:** Nội suy điền khuyết dữ liệu, chuẩn hóa (StandardScaler), và tạo đặc trưng chu kỳ thời gian (Sines/Cosines) để tối ưu hóa cho mô hình AI.
- **Dự báo Kép (Dual Inference):** - Mô hình **Random Forest**: Thích ứng nhanh, dự báo bám sát thực tế với lượng dữ liệu nhỏ.
  - Mô hình **LSTM (Mạng nơ-ron học sâu)**: Khai thác chuỗi thời gian bằng các cổng trí nhớ dài hạn.
- **Bảo toàn Trạng thái (State Preservation):** Cơ chế lưu trữ tự động ghép nối (Left Join) lịch sử dự báo để vẽ quỹ đạo liên tục.
- **Dashboard Tương tác (Streamlit):** Trực quan hóa dữ liệu bằng biểu đồ động **Plotly** và bản đồ nhiệt không gian-thời gian **Folium**.

## 🏗️ Kiến trúc Hệ thống
Dự án được chia thành các kịch bản (scripts) chạy tuần tự:
1. `data_ingestion.py`: Gọi API, kéo dữ liệu thô về `raw_aqi_data.csv`.
2. `data_preprocessing.py`: Làm sạch, chuẩn hóa, tính toán AQI và lưu đè lên `processed_aqi_data.csv` (Có bảo tồn lịch sử dự báo).
3. `inference.py`: Gọi file `rf_model.pkl` và `lstm_aqi_model.h5` để dự báo giá trị tương lai và ghi vào dòng mới nhất.
4. `dashboard.py`: Giao diện Web hiển thị kết quả trực quan.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

Chúng tôi cung cấp 2 cách để chạy dự án này, từ dễ đến chuyên sâu.

### Cách 1: Chạy ngay trên Đám mây (Khuyên dùng - Không cần cài đặt)
Dự án đã được tích hợp sẵn cấu hình **DevContainers**. Bạn không cần máy tính cấu hình cao.
1. Nhấn phím `.` (dấu chấm) hoặc biểu tượng **Codespaces** trên giao diện GitHub.
2. Đợi khoảng 1-2 phút. Hệ thống sẽ tự động tạo máy ảo ảo, cài đặt toàn bộ thư viện trong `requirements.txt` và tự động bật trình duyệt Web hiển thị Dashboard.

### Cách 2: Chạy trên Máy tính cá nhân (Local)
Yêu cầu hệ thống đã cài đặt Python 3.11+.

**Bước 1: Tải mã nguồn về máy**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

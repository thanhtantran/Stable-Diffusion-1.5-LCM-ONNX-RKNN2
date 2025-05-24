# Stable-Diffusion-1.5-LCM-ONNX-RKNN2

Dự án này cung cấp một giao diện web đơn giản để chạy **Stable Diffusion 1.5 với LCM** sử dụng **ONNX** và **RKNN2**, được tối ưu hóa cho các thiết bị [Orange Pi](https://orangepi.vn) với frontend **Streamlit**.

Với ứng dụng này, bạn có thể:
- Nhập mô tả hình ảnh bằng văn bản qua giao diện web
- Tạo hình ảnh bằng mô hình Stable Diffusion được tối ưu hóa cục bộ
- Hiển thị hình ảnh ngay trên frontend theo thời gian thực

---

## 🚀 Tính năng

- Giao diện web thân thiện được xây dựng bằng Streamlit
- Hỗ trợ công cụ suy luận Rockchip RKNN2
- Nhẹ, dễ triển khai trên các thiết bị [Orange Pi](https://orangepi.vn (như Orange Pi 5, Orange Pi 5B, Orange Pi 5 MAX ...)

---

## 🛠️ Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone https://github.com/thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2.git
cd Stable-Diffusion-1.5-LCM-ONNX-RKNN2
```

### 2. Cài đặt Miniforge3
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 3. Kích hoạt môi trường conda cơ bản
```bash
source ~/miniforge3/bin/activate
```

### 4. Tạo và kích hoạt môi trường Python cho dự án
```bash
conda create -n stable-diffusion-lcm python=3.10
conda activate stable-diffusion-lcm
```

### 5. Cài đặt RKNN Toolkit Lite 2 (phiên bản 2.3.0)
```bash
pip install rknn-toolkit-lite2/rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 6. Sao chép thư viện thực thi RKNN vào hệ thống
```bash
sudo cp lib/librknnrt.so /usr/lib
```

### 7. Cài đặt các thư viện Python cần thiết
```bash
pip install -r requirements.txt
```

---

## ▶️ Chạy ứng dụng

```bash
streamlit run app.py
```

Sau đó mở trình duyệt và truy cập `http://localhost:8501` để sử dụng ứng dụng.

---

## 📁 Cấu trúc dự án

```
Stable-Diffusion-1.5-LCM-ONNX-RKNN2/
├── app.py                        # Điểm khởi chạy ứng dụng Streamlit
├── requirements.txt             # Danh sách thư viện Python
├── lib/librknnrt.so             # Thư viện chạy RKNN
├── rknn-toolkit-lite2/         # Bộ toolkit RKNN2
└── models/, utils/, etc.        # Các tập tin hỗ trợ khác
```

---

## 📌 Ghi chú

- Dự án này thiết kế dành riêng cho hệ thống ARM64 (ví dụ: RK3588).
- Đảm bảo bạn có đúng phiên bản `librknnrt.so` và phần cứng tương thích.

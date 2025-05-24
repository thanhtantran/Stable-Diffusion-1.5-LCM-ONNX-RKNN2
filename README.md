[🇻🇳 Xem bản tiếng Việt](README-VIE.md)

# Stable-Diffusion-1.5-LCM-ONNX-RKNN2

This project provides a simple web interface to run **Stable Diffusion 1.5 with LCM** using **ONNX** and **RKNN2**, optimized for edge devices with **Streamlit** frontend.

With this app, you can:
- Enter a text prompt via a web interface
- Generate an image using a locally optimized Stable Diffusion model
- View the generated image in real-time on the frontend

---

## 🚀 Features

- Web UI built with Streamlit for ease of use
- Supports Rockchip RKNN2 inference engine
- Lightweight and easy to deploy on ARM-based systems (e.g., Orange Pi, Rockchip devices)

---

## 🛠️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2.git
cd Stable-Diffusion-1.5-LCM-ONNX-RKNN2
```

### 2. Install Miniforge3
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 3. Activate base conda environment
```bash
source ~/miniforge3/bin/activate
```

### 4. Create and activate the project environment
```bash
conda create -n stable-diffusion-lcm python=3.10
conda activate stable-diffusion-lcm
```

### 5. Install RKNN Toolkit Lite 2 (v2.3.0)
```bash
pip install rknn-toolkit-lite2/rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 6. Copy the RKNN runtime shared library
```bash
sudo cp lib/librknnrt.so /usr/lib
```

### 7. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser and go to `http://localhost:8501` to use the app.

---

## 📁 Project Structure

```
Stable-Diffusion-1.5-LCM-ONNX-RKNN2/
├── app.py                        # Streamlit app entry point
├── requirements.txt             # Python dependencies
├── lib/librknnrt.so             # RKNN runtime shared object
├── rknn-toolkit-lite2/         # RKNN2 toolkit wheel
└── models/, utils/, etc.        # Your supporting files
```

---

## 📌 Notes

- This project is designed for ARM64 systems (e.g., RK3588).
- Ensure you have the correct version of `librknnrt.so` and compatible hardware.

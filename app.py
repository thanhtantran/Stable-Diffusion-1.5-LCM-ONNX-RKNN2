import streamlit as st
import subprocess
import os
import glob
import time
import re
from hf_model_downloader import download_model, get_model_path

def sanitize_filename(prompt):
    """Chuyển prompt thành tên file hợp lệ"""
    filename = re.sub(r'[^\w\s-]', '', prompt)
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename

def initialize_model():
    """Khởi tạo model với thông báo trực quan"""
    with st.status("📦 Đang khởi tạo model...", expanded=True) as status:
        st.write("Kiểm tra thư mục model...")
        time.sleep(0.5)
        
        try:
            model_path = download_model()
            st.success(f"✅ Model sẵn sàng tại: {model_path}")
            status.update(label="Khởi tạo thành công!", state="complete")
            return True
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            status.update(label="Khởi tạo thất bại", state="error")
            return False

def run_image_generation(num_steps, size, prompt):
    """Tạo ảnh từ model"""
    try:
        cmd = [
            "python", "./run_rknn-lcm.py",
            "-i", "./model",
            "-o", "./images",
            "--num-inference-steps", str(num_steps),
            "-s", size,
            "--prompt", prompt
        ]
        subprocess.run(cmd, check=True)
        
        prompt_dir = sanitize_filename(prompt)
        latest_image = max(
            glob.glob(f"./images/{prompt_dir}/*.png"),
            key=os.path.getctime
        )
        return latest_image, None
    except Exception as e:
        return None, str(e)

def main():
    st.set_page_config(
        page_title="Orange Pi Image Generator",
        page_icon="🎨",
        layout="wide"
    )
    
    # Khởi tạo trạng thái
    if 'init' not in st.session_state:
        st.session_state.update({
            'init': False,
            'generating': False,
            'image_path': None,
            'error': None
        })
    
    # Tiêu đề
    st.title("🎨 Orange Pi AI Image Generator")
    st.markdown("""
    **Công cụ tạo ảnh chạy trên Orange Pi 5 (RK3588)**  
    [Mua Orange Pi tại đây](https://orangepi.net)
    """, unsafe_allow_html=True)
    
    # Khởi tạo model
    if not st.session_state.init:
        if initialize_model():
            st.session_state.init = True
            st.rerun()
        else:
            st.stop()
    
    # Giao diện chính
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Cài đặt")
        num_steps = st.slider("Số bước", 1, 4, 4)
        size = st.selectbox("Kích thước", ["384x384", "512x512"])
        prompt = st.text_area("Prompt", height=100, value="Phong cảnh núi tuyết đẹp")
        
        if st.button("Tạo ảnh"):
            st.session_state.generating = True
            st.session_state.image_path = None
            st.session_state.error = None
            st.rerun()
    
    with col2:
        st.header("Kết quả")
        
        if st.session_state.generating:
            with st.spinner("Đang tạo ảnh..."):
                progress = st.progress(0)
                for i in range(100):
                    progress.progress(i + 1)
                    time.sleep(0.03)
                
                image_path, error = run_image_generation(num_steps, size, prompt)
                
                if error:
                    st.session_state.error = error
                else:
                    st.session_state.image_path = image_path
                
                st.session_state.generating = False
                st.rerun()
        
        if st.session_state.error:
            st.error(f"Lỗi: {st.session_state.error}")
        
        if st.session_state.image_path:
            st.image(st.session_state.image_path, use_column_width=True)
            with open(st.session_state.image_path, "rb") as f:
                st.download_button(
                    "Tải ảnh",
                    f.read(),
                    file_name=os.path.basename(st.session_state.image_path),
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
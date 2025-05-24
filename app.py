import streamlit as st
import subprocess
import os
import glob
import time
from pathlib import Path
import re
from hf_model_downloader import download_model, get_model_path  # Thêm import thư viện tải model

def sanitize_filename(prompt):
    """Convert prompt to filename format (similar to your script's behavior)"""
    filename = re.sub(r'[^\w\s-]', '', prompt)
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename

def initialize_model():
    """Khởi tạo model và hiển thị trạng thái cho người dùng"""
    st.info("🔍 Đang kiểm tra thư mục model...")
    time.sleep(0.5)  # Hiệu ứng hiển thị
    
    try:
        model_path = download_model()
        st.success(f"✅ Model đã sẵn sàng tại: {model_path}")
        time.sleep(0.5)
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo model: {str(e)}")
        return False

def run_image_generation(num_steps, size, prompt):
    """Run the image generation command and return the output path"""
    try:
        cmd = [
            "python", "./run_rknn-lcm.py",
            "-i", "./model",
            "-o", "./images",
            "--num-inference-steps", str(num_steps),
            "-s", size,
            "--prompt", prompt
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        prompt_dir = sanitize_filename(prompt)
        search_pattern = f"./images/{prompt_dir}/*.png"
        time.sleep(1)
        
        image_files = glob.glob(search_pattern)
        if image_files:
            latest_file = max(image_files, key=os.path.getctime)
            return latest_file, None
        else:
            return None, "Image file not found after generation"
            
    except subprocess.CalledProcessError as e:
        return None, f"Command failed: {e.stderr}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Image Generator",
        page_icon="🎨",
        layout="wide"
    )
    
    # Khởi tạo session state
    if 'model_ready' not in st.session_state:
        st.session_state.model_ready = False
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # Hiển thị tiêu đề
    st.title("🎨 Local Orange Pi AI Image Generator")
    st.markdown("""
    Generate beautiful images using your RKNN-LCM model, using Orange Pi 5 with RK3588 SoC - Buy Orange Pi 5 RK3588 at <a href='https://orangepi.net' target='_blank'>https://orangepi.net</a>
    """, unsafe_allow_html=True)
    
    # Phần khởi tạo model
    if not st.session_state.model_ready:
        with st.status("📦 Đang chuẩn bị model...", expanded=True) as status:
            st.write("Kiểm tra và tải model nếu cần thiết")
            if initialize_model():
                status.update(label="✅ Model đã sẵn sàng!", state="complete")
                st.session_state.model_ready = True
            else:
                status.update(label="❌ Không thể khởi tạo model", state="error")
                return  # Dừng ứng dụng nếu không tải được model
    
    # Tạo giao diện chính sau khi model đã sẵn sàng
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Settings")
        
        num_steps = st.number_input(
            "Number of Inference Steps",
            min_value=1,
            max_value=4,
            value=4,
            disabled=st.session_state.generating
        )
        
        size_options = ["384x384", "512x512"]
        size = st.selectbox(
            "Image Size",
            options=size_options,
            index=1,
            disabled=st.session_state.generating
        )
        
        prompt = st.text_area(
            "Prompt",
            value="An astronaus rides a horse on Mars.",
            height=100,
            disabled=st.session_state.generating
        )
        
        if st.button("Generate Image", disabled=st.session_state.generating or not prompt.strip()):
            st.session_state.generating = True
            st.session_state.generated_image = None
            st.session_state.error_message = None
            st.rerun()
    
    with col2:
        st.header("Generated Image")
        
        if st.session_state.generating:
            st.info("🕐 Generating image... Please wait")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing... {i + 1}%")
                time.sleep(0.1)
            
            image_path, error = run_image_generation(num_steps, size, prompt)
            
            st.session_state.generating = False
            if error:
                st.session_state.error_message = error
            else:
                st.session_state.generated_image = image_path
            
            progress_bar.empty()
            status_text.empty()
            st.rerun()
        
        if st.session_state.error_message:
            st.error(f"❌ Error: {st.session_state.error_message}")
        
        if st.session_state.generated_image:
            if os.path.exists(st.session_state.generated_image):
                st.success("✅ Image generated successfully!")
                st.image(st.session_state.generated_image, caption="Generated Image", use_container_width=True)
                
                with open(st.session_state.generated_image, "rb") as file:
                    st.download_button(
                        label="📥 Download Image",
                        data=file.read(),
                        file_name=os.path.basename(st.session_state.generated_image),
                        mime="image/png"
                    )
            else:
                st.error("❌ Generated image file not found")
    
    st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
    .stButton > button:disabled {
        background-color: #CCCCCC;
        color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
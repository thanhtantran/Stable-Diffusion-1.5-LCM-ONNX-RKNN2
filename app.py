import streamlit as st
import subprocess
import os
import glob
import time
from pathlib import Path
import re
from hf_model_downloader import download_model  # Added model downloader

def sanitize_filename(prompt):
    """Convert prompt to filename format"""
    filename = re.sub(r'[^\w\s-]', '', prompt)
    filename = re.sub(r'[-\s]+', '_', filename)
    return filename

def initialize_model():
    """Khởi tạo model với feedback tối giản"""
    try:
        # Hiển thị thông báo cực ngắn nếu cần tải
        if not os.path.exists("./model") or not os.listdir("./model"):
            with st.spinner("Downloading model files..."):
                download_model()
        return True
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        return False

def run_image_generation(num_steps, size, prompt):
    """Run image generation command"""
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
        page_title="AI Image Generator",
        page_icon="🎨",
        layout="wide"
    )
    
    # Initialize session state
    if 'model_ready' not in st.session_state:
        st.session_state.model_ready = False
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    # Khởi tạo model (ẩn với người dùng)
    if 'model_ready' not in st.session_state:
        if not initialize_model():
            st.stop()
        st.session_state.model_ready = True
        st.rerun()
    
    # Main UI (identical to original)
    st.title("🎨 Local Orange Pi AI Image Generator")
    st.markdown("""
    Generate beautiful images using your RKNN-LCM model, using Orange Pi 5 with RK3588 SoC - Buy Orange Pi 5 RK3588 at <a href='https://orangepi.net' target='_blank'>https://orangepi.net</a>
    """, unsafe_allow_html=True)
    
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
            value="An astronaut rides a horse on Mars.",
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
                st.image(st.session_state.generated_image, 
                        caption="Generated Image", 
                        use_container_width=True)
                
                with open(st.session_state.generated_image, "rb") as file:
                    st.download_button(
                        label="📥 Download Image",
                        data=file.read(),
                        file_name=os.path.basename(st.session_state.generated_image),
                        mime="image/png"
                    )
            else:
                st.error("❌ Generated image file not found")
    
    # Original CSS styling
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
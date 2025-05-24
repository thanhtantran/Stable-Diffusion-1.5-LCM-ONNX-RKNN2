import streamlit as st
import subprocess
import os
import glob
import time
from pathlib import Path
import re
from hf_model_downloader import download_model

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

def run_image_generation(num_steps, size, prompt, progress_callback=None):
  """Run image generation command with real-time progress"""
  try:
      # Create output directory if it doesn't exist
      os.makedirs("./images", exist_ok=True)
      
      cmd = [
          "python", "./run_rknn-lcm.py",
          "-i", "./model",
          "-o", "./images",
          "--num-inference-steps", str(num_steps),
          "-s", size,
          "--prompt", prompt
      ]
      
      if progress_callback:
          progress_callback(0.1, "Starting image generation...")
      
      # Run the command
      process = subprocess.Popen(
          cmd, 
          stdout=subprocess.PIPE, 
          stderr=subprocess.PIPE, 
          text=True,
          universal_newlines=True
      )
      
      if progress_callback:
          progress_callback(0.3, "Model processing...")
      
      # Wait for process to complete
      stdout, stderr = process.communicate()
      
      if progress_callback:
          progress_callback(0.7, "Finalizing image...")
      
      if process.returncode != 0:
          return None, f"Command failed: {stderr}"
      
      if progress_callback:
          progress_callback(0.9, "Searching for generated image...")
      
      # Wait a moment for file to be written
      time.sleep(2)
      
      # Try multiple search patterns
      prompt_dir = sanitize_filename(prompt)
      search_patterns = [
          f"./images/{prompt_dir}/*.png",
          f"./images/{prompt_dir}/*.jpg",
          f"./images/*.png",
          f"./images/*.jpg"
      ]
      
      image_files = []
      for pattern in search_patterns:
          files = glob.glob(pattern)
          if files:
              image_files.extend(files)
              break
      
      if progress_callback:
          progress_callback(1.0, "Complete!")
      
      if image_files:
          # Return the most recently created file
          latest_image = max(image_files, key=os.path.getctime)
          return latest_image, None
      else:
          # Debug: List all files in images directory
          all_files = []
          if os.path.exists("./images"):
              for root, dirs, files in os.walk("./images"):
                  for file in files:
                      all_files.append(os.path.join(root, file))
          
          error_msg = f"No image files found. Searched patterns: {search_patterns}"
          if all_files:
              error_msg += f"\nFound files: {all_files}"
          else:
              error_msg += "\nNo files found in ./images directory"
          
          return None, error_msg
          
  except Exception as e:
      return None, f"Error: {str(e)}"

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
  
  # Khởi tạo model một lần duy nhất
  if not st.session_state.model_ready:
      if not initialize_model():
          st.stop()
      st.session_state.model_ready = True
      st.rerun()
  
  # Main UI
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
          # Create progress tracking
          progress_bar = st.progress(0)
          status_text = st.empty()
          
          def update_progress(progress, message):
              progress_bar.progress(progress)
              status_text.text(message)
          
          # Run the actual generation with real progress
          image_path, error = run_image_generation(num_steps, size, prompt, update_progress)
          
          # Update session state
          st.session_state.generating = False
          if error:
              st.session_state.error_message = error
          else:
              st.session_state.generated_image = image_path
          
          # Clear progress indicators and rerun
          progress_bar.empty()
          status_text.empty()
          st.rerun()
      
      # Display results
      if st.session_state.error_message:
          st.error(f"❌ Error: {st.session_state.error_message}")
          
          # Add debug button
          if st.button("🔍 Debug: Show files in images directory"):
              if os.path.exists("./images"):
                  all_files = []
                  for root, dirs, files in os.walk("./images"):
                      for file in files:
                          all_files.append(os.path.join(root, file))
                  if all_files:
                      st.write("Files found:")
                      for file in all_files:
                          st.write(f"- {file}")
                  else:
                      st.write("No files found in ./images directory")
              else:
                  st.write("./images directory does not exist")
      
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
  
  # CSS styling
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
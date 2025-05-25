import streamlit as st
import subprocess
import os
import glob
import time
from pathlib import Path
import re
from datetime import datetime
from hf_model_downloader import download_model
import base64

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

def get_base64_of_bin_file(bin_file):
  """Convert image to base64 string"""
  with open(bin_file, 'rb') as f:
      data = f.read()
  return base64.b64encode(data).decode()

def sanitize_filename(prompt):
  """Convert prompt to filename format (matching the actual script behavior)"""
  # This should match exactly how your run_rknn-lcm.py creates folder names
  filename = re.sub(r'[^\w\s-]', '', prompt)
  filename = re.sub(r'[-\s]+', '_', filename)
  # Remove trailing dots if any
  filename = filename.rstrip('.')
  return filename

def get_image_history():
  """Get all generated images sorted by creation time (newest first)"""
  try:
      all_images = glob.glob("./images/**/*.png", recursive=True)
      all_images.extend(glob.glob("./images/**/*.jpg", recursive=True))
      
      # Sort by creation time (newest first)
      all_images.sort(key=os.path.getctime, reverse=True)
      
      # Get file info
      image_info = []
      for img_path in all_images:
          try:
              stat = os.stat(img_path)
              creation_time = datetime.fromtimestamp(stat.st_ctime)
              file_size = stat.st_size
              
              image_info.append({
                  'path': img_path,
                  'filename': os.path.basename(img_path),
                  'creation_time': creation_time,
                  'size': file_size,
                  'size_mb': round(file_size / (1024 * 1024), 2)
              })
          except:
              continue
      
      return image_info
  except Exception as e:
      return []

def run_image_generation(num_steps, size, prompt):
  """Run the image generation command and return the output path"""
  try:
      # Construct the command
      cmd = [
          "python", "./run_rknn-lcm.py",
          "-i", "./model",
          "-o", "./images",
          "--num-inference-steps", str(num_steps),
          "-s", size,
          "--prompt", prompt
      ]
      
      # Run the command
      result = subprocess.run(cmd, capture_output=True, text=True, check=True)
      
      # Wait a moment for file to be written
      time.sleep(2)
      
      # Try multiple approaches to find the generated image
      prompt_dir = sanitize_filename(prompt)
      
      # Search patterns - try different variations
      search_patterns = [
          f"./images/{prompt_dir}/*.png",
          f"./images/{prompt_dir}./*.png",  # With dot at the end
          f"./images/{prompt_dir}*/*.png",  # Wildcard matching
          f"./images/*{prompt_dir}*/*.png",  # More flexible matching
          f"./images/**/*.png",  # Search all subdirectories
      ]
      
      image_files = []
      for pattern in search_patterns:
          files = glob.glob(pattern, recursive=True)
          if files:
              image_files.extend(files)
              break
      
      if image_files:
          # Return the most recently created file
          latest_file = max(image_files, key=os.path.getctime)
          return latest_file, None
      else:
          # If still not found, search all PNG files and find the newest
          all_pngs = glob.glob("./images/**/*.png", recursive=True)
          if all_pngs:
              # Get the most recent PNG file
              latest_file = max(all_pngs, key=os.path.getctime)
              return latest_file, None
          else:
              # Debug info
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
          
  except subprocess.CalledProcessError as e:
      return None, f"Command failed: {e.stderr}"
  except Exception as e:
      return None, f"Error: {str(e)}"

def main():
  st.set_page_config(
      page_title="Local Orange Pi AI Image Generator",
      page_icon="🎨",
      layout="wide"
  )
  
  # Header with logo
  header_col1, header_col2 = st.columns([3, 1])
  
  with header_col1:
      st.title("🎨 Local Orange Pi AI Image Generator")
      st.markdown("""
      Generate beautiful images using your RKNN-LCM model, using Orange Pi 5 with RK3588 SoC - Buy Orange Pi 5 RK3588 at <a href='https://orangepi.net' target='_blank'>https://orangepi.net</a>
      """, unsafe_allow_html=True)
  
  with header_col2:
      # Display logo if it exists
      logo_path = "logo1.png"
      if os.path.exists(logo_path):
          st.image(logo_path, width=200)
      else:
          # Fallback if logo file not found
          st.markdown("""
          <div style='text-align: center; padding: 20px;'>
              <a href='https://orangepi.vn' target='_blank'>
                  <img src='https://d41chssnpqdne.cloudfront.net/user_upload_by_module/chat_bot/files/11649934/yk5DLMH5wgC8lGkp.png' width='200' alt='Orange Pi Vietnam'>
              </a>
          </div>
          """, unsafe_allow_html=True)
  
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
  
  # Create two columns for layout
  col1, col2 = st.columns([1, 1])
  
  with col1:
      st.header("Settings")
      
      # Input fields - disabled when generating
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
          index=1,  # Default to 512x512
          disabled=st.session_state.generating
      )
      
      prompt = st.text_area(
          "Prompt",
          value="An astronaut rides a horse on Mars.",
          height=100,
          disabled=st.session_state.generating
      )
      
      # Generate button
      if st.button("Generate Image", disabled=st.session_state.generating or not prompt.strip()):
          st.session_state.generating = True
          st.session_state.generated_image = None
          st.session_state.error_message = None
          st.rerun()
      
      # Image History Section
      st.markdown("---")
      st.header("📁 Image History")
      
      # Get image history
      image_history = get_image_history()
      
      if image_history:
          st.write(f"**Total images generated:** {len(image_history)}")
          
          # Show images in expandable sections
          for i, img_info in enumerate(image_history[:10]):  # Show latest 10 images
              with st.expander(f"🖼️ {img_info['filename']} ({img_info['creation_time'].strftime('%Y-%m-%d %H:%M:%S')})"):
                  col_img, col_info = st.columns([1, 1])
                  
                  with col_img:
                      if os.path.exists(img_info['path']):
                          st.image(img_info['path'], width=200)
                  
                  with col_info:
                      st.write(f"**Created:** {img_info['creation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                      st.write(f"**Size:** {img_info['size_mb']} MB")
                      st.write(f"**Path:** `{img_info['path']}`")
                      
                      # Download button for each image
                      if os.path.exists(img_info['path']):
                          with open(img_info['path'], "rb") as file:
                              st.download_button(
                                  label="📥 Download",
                                  data=file.read(),
                                  file_name=img_info['filename'],
                                  mime="image/png",
                                  key=f"download_{i}"
                              )
          
          if len(image_history) > 10:
              st.info(f"Showing latest 10 images. Total: {len(image_history)} images generated.")
          
          # Bulk actions
          st.markdown("---")
          col_refresh, col_clear = st.columns(2)
          
          with col_refresh:
              if st.button("🔄 Refresh History"):
                  st.rerun()
          
          with col_clear:
              if st.button("🗑️ Clear All Images", type="secondary"):
                  if st.session_state.get('confirm_clear', False):
                      try:
                          # Delete all images
                          for img_info in image_history:
                              if os.path.exists(img_info['path']):
                                  os.remove(img_info['path'])
                          
                          # Remove empty directories
                          for root, dirs, files in os.walk("./images", topdown=False):
                              for dir_name in dirs:
                                  dir_path = os.path.join(root, dir_name)
                                  try:
                                      if not os.listdir(dir_path):
                                          os.rmdir(dir_path)
                                  except:
                                      pass
                          
                          st.session_state.confirm_clear = False
                          st.success("All images cleared!")
                          st.rerun()
                      except Exception as e:
                          st.error(f"Error clearing images: {e}")
                  else:
                      st.session_state.confirm_clear = True
                      st.warning("Click again to confirm deletion of all images")
      else:
          st.info("No images generated yet. Create your first image!")
  
  with col2:
      st.header("Generated Image")
      
      if st.session_state.generating:
          # Show loading state
          st.info("🕐 Generating image... Please wait")
          
          # Create a progress bar
          progress_bar = st.progress(0)
          status_text = st.empty()
          
          # Simulate progress (since we can't get real progress from the subprocess)
          for i in range(100):
              progress_bar.progress(i + 1)
              status_text.text(f"Processing... {i + 1}%")
              time.sleep(0.05)  # Faster progress
          
          # Run the actual generation
          image_path, error = run_image_generation(num_steps, size, prompt)
          
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
          
          # Add debug button to manually find the latest image
          if st.button("🔍 Try to find latest generated image"):
              try:
                  all_pngs = glob.glob("./images/**/*.png", recursive=True)
                  if all_pngs:
                      latest_file = max(all_pngs, key=os.path.getctime)
                      st.session_state.generated_image = latest_file
                      st.session_state.error_message = None
                      st.rerun()
                  else:
                      st.write("No PNG files found in images directory")
              except Exception as e:
                  st.write(f"Error finding files: {e}")
      
      if st.session_state.generated_image:
          if os.path.exists(st.session_state.generated_image):
              st.success("✅ Image generated successfully!")
              st.image(st.session_state.generated_image, caption="Generated Image", use_container_width=True)
              
              # Add download button
              with open(st.session_state.generated_image, "rb") as file:
                  st.download_button(
                      label="📥 Download Image",
                      data=file.read(),
                      file_name=os.path.basename(st.session_state.generated_image),
                      mime="image/png"
                  )
          else:
              st.error("❌ Generated image file not found")
  
  # Footer
  st.markdown("---")
  st.markdown("""
  <div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>
      <p>Copyright © 2025 - <a href='https://orangepi.vn' target='_blank' style='color: #FF6B6B; text-decoration: none;'>Orange Pi Việt Nam</a></p>
      <p>Powered by Orange Pi 5 RK3588 SoC | AI Image Generation with RKNN-LCM</p>
  </div>
  """, unsafe_allow_html=True)
  
  # Add some styling
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
  .stExpander > div > div > div > div {
      padding: 10px;
  }
  
  /* Header styling */
  .main > div:first-child {
      padding-top: 1rem;
  }
  
  /* Footer styling */
  .footer {
      position: fixed;
      left: 0;
      bottom: 0;
      width: 100%;
      background-color: #f0f2f6;
      color: #666;
      text-align: center;
      padding: 10px;
      border-top: 1px solid #e0e0e0;
  }
  
  /* Logo styling */
  .logo-container {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      padding: 10px;
  }
  </style>
  """, unsafe_allow_html=True)

if __name__ == "__main__":
  main()
import os
from huggingface_hub import snapshot_download

REPO_ID = "thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2"
MODEL_FOLDER = "model"
LOCAL_MODEL_DIR = "./model"  # Tải thẳng vào ./model

def download_model(force_redownload=False):
    """
    Tải folder 'model' từ Hugging Face Hub nếu chưa tồn tại cục bộ.
    
    Args:
        force_redownload (bool): Nếu True, tải lại dù folder đã tồn tại.
    
    Returns:
        str: Đường dẫn tuyệt đối đến folder model.
    """
    # Kiểm tra và tạo thư mục nếu cần
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    if not force_redownload and os.path.exists(os.path.join(LOCAL_MODEL_DIR, ".gitattributes")):
        print("✅ Folder model đã tồn tại. Bỏ qua tải xuống.")
        return os.path.abspath(LOCAL_MODEL_DIR)

    print("🔄 Đang tải folder 'model' từ Hugging Face Hub...")
    try:
        # Tải chỉ nội dung trong folder 'model' của repo
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{MODEL_FOLDER}/*",
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ Đã tải xong vào: {os.path.abspath(LOCAL_MODEL_DIR)}")
        return os.path.abspath(LOCAL_MODEL_DIR)
    
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        raise

def get_model_path():
    """Trả về đường dẫn tuyệt đối đến folder model (không kiểm tra tồn tại)."""
    return os.path.abspath(LOCAL_MODEL_DIR)
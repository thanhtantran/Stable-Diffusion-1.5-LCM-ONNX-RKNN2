import os
from huggingface_hub import snapshot_download

REPO_ID = "thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2"
MODEL_FOLDER = "model"
LOCAL_MODEL_DIR = "./model"  # Thư mục đích

def download_model(force_redownload=False):
    """
    Tải tất cả file từ folder model của repo vào LOCAL_MODEL_DIR
    (Không tạo thư mục con model bên trong)
    """
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    # Kiểm tra nếu model đã tồn tại
    if not force_redownload and os.listdir(LOCAL_MODEL_DIR):
        return os.path.abspath(LOCAL_MODEL_DIR)

    try:
        # Tải thẳng vào LOCAL_MODEL_DIR (không tạo thư mục con)
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{MODEL_FOLDER}/*",
            local_dir=LOCAL_MODEL_DIR,
            local_dir_root=".",  # Quan trọng: chỉ định root directory
            force_download=force_redownload,
        )
        return os.path.abspath(LOCAL_MODEL_DIR)
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def get_model_path():
    """Trả về đường dẫn tuyệt đối đến thư mục model"""
    return os.path.abspath(LOCAL_MODEL_DIR)
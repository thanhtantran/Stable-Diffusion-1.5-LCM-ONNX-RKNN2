import os
from huggingface_hub import snapshot_download

REPO_ID = "thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2"
MODEL_FOLDER = "model"
LOCAL_MODEL_DIR = "./"

def download_model(force_redownload=False):
    """Tải model với API mới nhất của Hugging Face Hub"""
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    # Kiểm tra nếu model đã tồn tại
    if not force_redownload and os.path.exists(os.path.join(LOCAL_MODEL_DIR, ".gitattributes")):
        return os.path.abspath(LOCAL_MODEL_DIR)

    try:
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{MODEL_FOLDER}/*",
            local_dir=LOCAL_MODEL_DIR,
            force_download=force_redownload,
        )
        return os.path.abspath(LOCAL_MODEL_DIR)
    except Exception as e:
        raise Exception(f"Lỗi tải model: {str(e)}")

def get_model_path():
    """Lấy đường dẫn tuyệt đối đến folder model"""
    return os.path.abspath(LOCAL_MODEL_DIR)
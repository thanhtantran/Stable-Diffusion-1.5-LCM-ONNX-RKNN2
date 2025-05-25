import os
import shutil
from huggingface_hub import snapshot_download

REPO_ID = "thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2"
MODEL_FOLDER = "model"
LOCAL_MODEL_DIR = "./model"  # Thư mục đích

def download_model(force_redownload=False):
    """
    Tải tất cả file từ folder model của repo vào LOCAL_MODEL_DIR
    (Không tạo thư mục con model bên trong)
    """
    # Kiểm tra nếu model đã tồn tại
    if not force_redownload and os.path.exists(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR):
        return os.path.abspath(LOCAL_MODEL_DIR)
    
    try:
        # Xóa thư mục cũ nếu force_redownload
        if force_redownload and os.path.exists(LOCAL_MODEL_DIR):
            shutil.rmtree(LOCAL_MODEL_DIR)
        
        # Tải trực tiếp vào thư mục cha, sau đó đổi tên
        parent_dir = os.path.dirname(LOCAL_MODEL_DIR)
        temp_name = LOCAL_MODEL_DIR + "_temp"
        
        # Tải về với tên tạm thời
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{MODEL_FOLDER}/*",
            local_dir=temp_name,
            force_download=force_redownload,
        )
        
        # Đổi tên từ temp_name/model thành LOCAL_MODEL_DIR
        downloaded_model_path = os.path.join(temp_name, MODEL_FOLDER)
        if os.path.exists(downloaded_model_path):
            shutil.move(downloaded_model_path, LOCAL_MODEL_DIR)
            # Xóa thư mục tạm thời
            if os.path.exists(temp_name):
                shutil.rmtree(temp_name)
        else:
            raise Exception("Model folder not found in downloaded content")
            
        return os.path.abspath(LOCAL_MODEL_DIR)
        
    except Exception as e:
        # Dọn dẹp nếu có lỗi
        temp_name = LOCAL_MODEL_DIR + "_temp"
        if os.path.exists(temp_name):
            shutil.rmtree(temp_name)
        raise Exception(f"Download failed: {str(e)}")

def get_model_path():
    """Trả về đường dẫn tuyệt đối đến thư mục model"""
    return os.path.abspath(LOCAL_MODEL_DIR)
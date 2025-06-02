import os
import shutil
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn tới tập dữ liệu gốc
tess_dir = "./TESS"  # Thay bằng đường dẫn thực tế tới TESS
crema_dir = "./CREMA-D"  # Thay bằng đường dẫn thực tế tới CREMA-D
output_dir = "merged_dataset"  # Thư mục đầu ra

# Danh sách 6 nhãn cảm xúc
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Ánh xạ mã cảm xúc CREMA-D sang nhãn
emotion_codes = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# Tạo thư mục đầu ra
for label in labels:
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    logger.info(f"Created directory: {label_dir}")

# Xử lý TESS
for root, _, files in os.walk(tess_dir):
    for file in files:
        if file.endswith(".wav"):
            # Lấy nhãn từ tên thư mục
            folder_name = os.path.basename(root).lower()
            # Tách phần cảm xúc (bỏ YAF_ hoặc OAF_)
            if folder_name.startswith("yaf_") or folder_name.startswith("oaf_"):
                label = folder_name[4:]  # Bỏ tiền tố YAF_ hoặc OAF_
            else:
                label = folder_name
            
            # Hợp nhất pleasant_surprise vào happy
            if "pleasant_surprise" in label:
                label = "happy"
            
            # Chỉ xử lý các nhãn thuộc danh sách mong muốn
            if label in labels:
                src_path = os.path.join(root, file)
                # Đảm bảo tên file không trùng lặp
                base_name = os.path.basename(file)
                dst_path = os.path.join(output_dir, label, f"tess_{base_name}")
                try:
                    shutil.copy(src_path, dst_path)
                    logger.info(f"Copied TESS file: {src_path} -> {dst_path}")
                except Exception as e:
                    logger.error(f"Error copying TESS file {src_path}: {e}")
            else:
                logger.warning(f"Skipping TESS file with label {label}: {file}")

# Xử lý CREMA-D
for file in os.listdir(crema_dir):
    if file.endswith(".wav"):
        parts = file.split("_")
        if len(parts) >= 3 and parts[2] in emotion_codes:
            label = emotion_codes[parts[2]]
            src_path = os.path.join(crema_dir, file)
            # Đảm bảo tên file không trùng lặp
            dst_path = os.path.join(output_dir, label, f"crema_{file}")
            try:
                shutil.copy(src_path, dst_path)
                logger.info(f"Copied CREMA-D file: {src_path} -> {dst_path}")
            except Exception as e:
                logger.error(f"Error copying CREMA-D file {src_path}: {e}")
        else:
            logger.warning(f"Skipping CREMA-D file with invalid format: {file}")

logger.info("Data merging completed!")
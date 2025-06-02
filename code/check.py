import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Đường dẫn tới tập dữ liệu hợp nhất
merged_dir = "merged_dataset"  # Thay bằng đường dẫn thực tế nếu khác

# Danh sách nhãn cảm xúc
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Hàm đếm file WAV trong một thư mục
def count_wav_files(directory):
    if not os.path.exists(directory):
        logger.error(f"Thư mục {directory} không tồn tại!")
        return 0
    count = 0
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            count += 1
    return count

# Đếm file cho từng nhãn và tổng cộng
total_files = 0
for label in labels:
    label_dir = os.path.join(merged_dir, label)
    if os.path.exists(label_dir):
        num_files = count_wav_files(label_dir)
        logger.info(f"Nhãn {label}: {num_files} file")
        total_files += num_files
    else:
        logger.warning(f"Thư mục {label_dir} không tồn tại!")

logger.info(f"Tổng số file WAV: {total_files}")
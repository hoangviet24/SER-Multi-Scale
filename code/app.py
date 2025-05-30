import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from predict import predict_emotion, MSTRModel, EmotionAudioDataset
from PIL import Image, ImageTk
import os
import pygame
import threading
import time
from utils import convert_to_wav
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    pygame.mixer.init()
except pygame.error as e:
    logger.error(f"pygame mixer init failed: {e}")
    pygame.mixer = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_tess = EmotionAudioDataset('./TESS', max_len=100)

model_tess = MSTRModel(
    input_dim=40,
    num_classes=len(dataset_tess.label_map),
    num_scales=3,
    window_size=4,
    num_heads=4
).to(device)

try:
    if os.path.exists('./models/best_model.pth'):
        model_tess.load_state_dict(torch.load('./models/best_model.pth', map_location=device))
        logger.info("Loaded TESS model from ./models/best_model.pth")
    else:
        messagebox.showwarning("Cảnh báo", "Không tìm thấy mô hình TESS. Sẽ dùng CREMA-D.")
except Exception as e:
    messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
    raise

root = tk.Tk()
root.title("Nhận Diện Cảm Xúc Qua Giọng Nói")
root.geometry("800x650")
root.configure(bg="#f0f0f0")
root.minsize(400, 300)

original_img = None
current_photo = None
last_valid_size = (800, 650)
current_audio_file = None
processed_audio = None
last_resize_time = 0
resize_cooldown = 0.4
last_image_size = (0, 0)
selected_model = tk.StringVar(value="tess")

# Add a label to display the selected file name
file_name_label = tk.Label(root, text="No file selected", font=("Arial", 12), bg="#f0f0f0")

def resize_image(event):
    global original_img, current_photo, last_valid_size, last_resize_time, last_image_size
    current_time = time.time()
    if current_time - last_resize_time < resize_cooldown:
        return
    last_resize_time = current_time

    if original_img is None:
        return
    
    if event.width < 400 or event.height < 300 or root.state() != 'normal':
        return
    
    window_width = event.width
    window_height = event.height
    if (window_width, window_height) == last_valid_size:
        return
    
    last_valid_size = (window_width, window_height)
    logger.info(f"Resize event: width={window_width}, height={window_height}")
    
    img_width, img_height = original_img.size
    aspect_ratio = img_width / img_height
    max_width = int(window_width * 0.7)
    max_height = int(window_height * 0.7)
    
    if max_width / aspect_ratio <= max_height:
        new_width = max(300, max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max(200, max_height)
        new_width = int(new_height * aspect_ratio)
    
    new_width = min(new_width, 800)
    new_height = min(new_height, 600)
    
    if (new_width, new_height) == last_image_size:
        return
    
    try:
        resized_img = original_img.resize((new_width, new_height), Image.BILINEAR)
        current_photo = ImageTk.PhotoImage(resized_img)
        image_label.config(image=current_photo)
        image_label.image = current_photo
        last_image_size = (new_width, new_height)
        logger.info(f"Resized image to: width={new_width}, height={new_height}")
    except Exception as e:
        logger.error(f"Resize error: {e}")

def upload_file():
    global original_img, current_photo, last_valid_size, current_audio_file, last_image_size, processed_audio
    file_path = filedialog.askopenfilename(filetypes=[("Audio/Video files", "*.wav *.mp3 *.ogg *.flac *.mp4")])
    if not file_path:
        return
    
    # Update the file name label
    file_name = os.path.basename(file_path)
    file_name_label.config(text=f"Selected file: {file_name}")
    
    temp_dir = os.path.abspath("./temp")
    os.makedirs(temp_dir, exist_ok=True)
    current_audio_file = os.path.join(temp_dir, file_name)
    try:
        with open(current_audio_file, 'wb') as f:
            with open(file_path, 'rb') as src:
                f.write(src.read())
        logger.info(f"Saved uploaded file to {current_audio_file}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")
        return
    
    processed_audio = None
    result_label.config(text="Đang xử lý...", font=("Arial", 14, "bold"))
    root.update()

    def process_audio():
        global original_img, current_photo, last_image_size, processed_audio, current_audio_file, last_valid_size
        try:
            processed_audio = convert_to_wav(current_audio_file, temp_dir=temp_dir)
            model = model_tess
            dataset = dataset_tess
            predicted_emotion, confidence = predict_emotion(
                processed_audio,
                model,
                dataset.label_map,
                device,
                max_len=100,
                confidence_threshold=50,
                cache_dir=os.path.abspath("./app_mfcc_cache")  # Sử dụng thư mục riêng cho app
            )
            root.after(0, lambda: result_label.config(text=f"Cảm xúc: {predicted_emotion} ({confidence:.2f}%)", font=("Arial", 14, "bold")))
            
            original_img = Image.open("emotion_probabilities.png")
            window_width = root.winfo_width()
            window_height = root.winfo_height()
            last_valid_size = (window_width, window_height)
            aspect_ratio = original_img.size[0] / original_img.size[1]
            max_width = int(window_width * 0.7)
            max_height = int(window_height * 0.7)
            if max_width / aspect_ratio <= max_height:
                new_width = max(300, max_width)
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max(200, max_height)
                new_width = int(new_height * aspect_ratio)
            
            new_width = min(new_width, 800)
            new_height = min(new_height, 600)
            
            resized_img = original_img.resize((new_width, new_height), Image.BILINEAR)
            current_photo = ImageTk.PhotoImage(resized_img)
            root.after(0, lambda: image_label.config(image=current_photo))
            root.after(0, lambda: setattr(image_label, 'image', current_photo))
            last_image_size = (new_width, new_height)
            logger.info(f"Initial image size: width={new_width}, height={new_height}")
            
            root.after(0, lambda: play_button.config(state='normal'))
            root.after(0, lambda: stop_button.config(state='disabled'))
            root.after(0, lambda: progress_label.config(text="Playing: 0.0s"))
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            root.after(0, lambda: messagebox.showerror("Lỗi", f"Không thể xử lý file: {str(e)}"))
            root.after(0, lambda: result_label.config(text="Chưa có kết quả"))
            current_audio_file = None
            processed_audio = None
            root.after(0, lambda: play_button.config(state='disabled'))
            root.after(0, lambda: stop_button.config(state='disabled'))
            root.after(0, lambda: progress_label.config(text="Playing: 0.0s"))
            # Reset file name label on error
            root.after(0, lambda: file_name_label.config(text="No file selected"))
        finally:
            pass

    threading.Thread(target=process_audio, daemon=True).start()

def play_audio():
    global processed_audio
    if processed_audio and os.path.exists(processed_audio) and pygame.mixer:
        try:
            pygame.mixer.music.load(processed_audio)
            pygame.mixer.music.play()
            play_button.config(state='disabled')
            stop_button.config(state='normal')
            update_progress()
        except pygame.error as e:
            logger.error(f"Error playing audio: {e}")
            messagebox.showerror("Lỗi", f"Không thể phát âm thanh: {str(e)}")
    else:
        messagebox.showerror("Lỗi", "Vui lòng upload file hợp lệ!")

def stop_audio():
    global processed_audio
    if pygame.mixer:
        pygame.mixer.music.stop()
    play_button.config(state='normal')
    stop_button.config(state='disabled')
    progress_label.config(text="Playing: 0.0s")
    if processed_audio and os.path.exists(processed_audio):
        try:
            os.remove(processed_audio)
            logger.info(f"Đã xóa file tạm: {processed_audio}")
        except Exception as e:
            logger.error(f"Error deleting temporary file: {e}")

def update_progress():
    if pygame.mixer and pygame.mixer.music.get_busy():
        pos = pygame.mixer.music.get_pos() / 1000
        progress_label.config(text=f"Playing: {pos:.1f}s")
        root.after(100, update_progress)
    else:
        progress_label.config(text="Playing: 0.0s")

def clear_temp_folder():
    """Xóa tất cả file trong thư mục ./temp"""
    temp_dir = os.path.abspath("./temp")
    if os.path.exists(temp_dir):
        try:
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted file in temp folder: {file_path}")
            logger.info(f"Cleared all files in {temp_dir}")
        except Exception as e:
            logger.error(f"Error clearing temp folder {temp_dir}: {e}")
    else:
        logger.info(f"Temp folder {temp_dir} does not exist, no files to clear")

def on_closing():
    """Xử lý sự kiện khi đóng cửa sổ"""
    try:
        if pygame.mixer:
            pygame.mixer.music.stop()
            logger.info("Stopped pygame mixer")
        clear_temp_folder()
    except Exception as e:
        logger.error(f"Error during closing: {e}")
    finally:
        root.destroy()

title_label = tk.Label(root, text="Nhận Diện Cảm Xúc", font=("Arial", 20, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

model_frame = tk.Frame(root, bg="#f0f0f0")
model_frame.pack(pady=5)

upload_button = tk.Button(root, text="Upload File Âm Thanh/Video", command=upload_file, font=("Arial", 12), bg="#4CAF50", fg="white", relief="flat", padx=15, pady=8)
upload_button.pack(pady=5)

# Place the file name label below the upload button
file_name_label.pack(pady=5)

button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=5)

play_button = tk.Button(button_frame, text="Play Audio", command=play_audio, font=("Arial", 12), bg="#2196F3", fg="white", relief="flat", padx=15, pady=8, state='disabled')
play_button.pack(side=tk.LEFT, padx=5)

stop_button = tk.Button(button_frame, text="Stop Audio", command=stop_audio, font=("Arial", 12), bg="#F44336", fg="white", relief="flat", padx=15, pady=8, state='disabled')
stop_button.pack(side=tk.LEFT, padx=5)

result_label = tk.Label(root, text="Chưa có kết quả", font=("Arial", 12), bg="#f0f0f0")
result_label.pack(pady=8)

progress_label = tk.Label(root, text="Playing: 0.0s", font=("Arial", 10), bg="#f0f0f0")
progress_label.pack(pady=5)

image_label = tk.Label(root, bg="#f0f0f0")
image_label.pack(pady=8, expand=True)

root.bind("<Configure>", resize_image)
root.protocol("WM_DELETE_WINDOW", on_closing)  # Gắn sự kiện đóng cửa sổ

root.mainloop()
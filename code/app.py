import tkinter as tk
from tkinter import Canvas, Scrollbar, ttk
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
import shutil
import wave

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    pygame.mixer.init()
except pygame.error as e:
    logger.error(f"pygame mixer init failed: {e}")
    pygame.mixer = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = EmotionAudioDataset('./TESS', max_len=100)

model = MSTRModel(
    input_dim=40,
    num_classes=len(dataset.label_map),
    num_scales=3,
    window_size=4,
    num_heads=4
).to(device)

try:
    model_path = './models/best_model_mfcc.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    else:
        messagebox.showwarning("Cảnh báo", f"Không tìm thấy mô hình tại {model_path}. Sẽ dùng mặc định.")
except Exception as e:
    messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
    raise

def get_audio_length(file_path):
    with wave.open(file_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration

# Tạo cửa sổ chính
root = tk.Tk()

# Biến toàn cục
original_img = None
current_photo = None
last_valid_size = (1065, 883)
current_audio_file = None
processed_audio = None
last_resize_time = 0
resize_cooldown = 0.4
last_image_size = (0, 0)
selected_model = tk.StringVar(value="TESS")

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
    file_name_label.config(text=f"Tên file: {file_name}")
    
    temp_dir = os.path.abspath("./temp")
    os.makedirs(temp_dir, exist_ok=True)
    current_audio_file = os.path.join(temp_dir, file_name)
    try:
        shutil.copyfile(file_path, current_audio_file)
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
            audio_length = get_audio_length(processed_audio)
            root.after(0, lambda: progress_bar.config(maximum=audio_length))
            predicted_emotion, confidence = predict_emotion(
                processed_audio,
                model,
                dataset.label_map,
                device,
                max_len=100,
                confidence_threshold=50,
                cache_dir=os.path.abspath("./app_mfcc_cache")
            )
            
            root.after(0, lambda: result_label.config(text=f"Cảm xúc: {predicted_emotion} ({confidence:.2f}%)", font=("Arial", 14, "bold")))
            
            if os.path.exists("emotion_probabilities.png"):
                original_img = Image.open("emotion_probabilities.png")
            else:
                logger.warning("emotion_probabilities.png not found")
                return

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
        progress_bar['value'] = pos
        root.after(100, update_progress)
    else:
        progress_label.config(text="Playing: 0.0s")
        progress_bar['value'] = 0
        play_button.config(state='normal')
        stop_button.config(state='disabled')

def clear_temp_folder():
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
    try:
        if pygame.mixer:
            pygame.mixer.music.stop()
            logger.info("Stopped pygame mixer")
        clear_temp_folder()
    except Exception as e:
        logger.error(f"Error during closing: {e}")
    finally:
        root.destroy()

def save_image():
    global original_img
    if original_img is None:
        messagebox.showwarning("Cảnh báo", "Chưa có hình ảnh nào để lưu!")
        return
    filetypes = [("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
    if save_path:
        try:
            original_img.save(save_path)
            messagebox.showinfo("Thành công", f"Đã lưu hình ảnh vào:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu hình ảnh: {str(e)}")

root.title("Nhận Diện Cảm Xúc Qua Giọng Nói")
root.geometry("600x400")
root.configure(bg="#f0f0f0")
root.minsize(400, 300)

# ======= Tạo canvas + scrollbar =======
main_canvas = tk.Canvas(root, bg="#f0f0f0", highlightthickness=0)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
scrollable_frame = tk.Frame(main_canvas, bg="#f0f0f0")

scrollable_frame.bind(
    "<Configure>",
    lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
)

main_canvas.create_window((0, 0), window=scrollable_frame, anchor="n", tags="frame")
main_canvas.configure(yscrollcommand=scrollbar.set)

main_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ======= Thêm các widget vào scrollable_frame =======
style = {"font": ("Arial", 12), "bg": "#ffffff", "fg": "#000000"}

save_image_button = tk.Button(scrollable_frame, text="💾 Lưu Hình Ảnh", command=save_image, **style)
save_image_button.pack(pady=10)

title_label = tk.Label(scrollable_frame, text="Nhận Diện Cảm Xúc", font=("Arial", 20, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

model_frame = tk.Frame(scrollable_frame, bg="#f0f0f0")
model_frame.pack(pady=5)

upload_button = tk.Button(scrollable_frame, text="🎵 Tải File Âm Thanh/Video", command=upload_file, **style)
upload_button.pack(pady=10)

file_name_label = tk.Label(scrollable_frame, text="Tên file...", font=("Arial", 12), bg="#f0f0f0")
file_name_label.pack(pady=5)

result_label = tk.Label(scrollable_frame, text="Chưa có kết quả", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

image_label = tk.Label(scrollable_frame, bg="#f0f0f0")
image_label.pack(pady=10)

control_frame = tk.Frame(scrollable_frame, bg="#f0f0f0")
control_frame.pack(pady=10)

play_button = tk.Button(control_frame, text="▶️ Phát Âm Thanh", command=play_audio, **style, state='disabled')
play_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(control_frame, text="⏹ Dừng Lại", command=stop_audio, **style, state='disabled')
stop_button.grid(row=0, column=1, padx=10)

progress_label = tk.Label(scrollable_frame, text="Playing: 0.0s", font=("Arial", 12), bg="#f0f0f0")
progress_label.pack(pady=5)

progress_bar = ttk.Progressbar(scrollable_frame, orient='horizontal', length=400, mode='determinate')
progress_bar.pack(pady=5)

# ======= Bind window resize & protocol =======
root.bind('<Configure>', resize_image)
root.protocol("WM_DELETE_WINDOW", root.quit)
def on_canvas_resize(event):
    canvas_width = event.width
    main_canvas.coords("frame", canvas_width // 2, 0)
    scrollable_frame.config(width=canvas_width)
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))

main_canvas.bind("<Configure>", on_canvas_resize)

main_canvas.configure(yscrollcommand=scrollbar.set)
main_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
root.mainloop()
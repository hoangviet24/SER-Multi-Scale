import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import Counter
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import tempfile
import uuid
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(file_path, temp_dir="./temp"):
    """Chuyển file không phải WAV thành WAV, trả về đường dẫn file WAV"""
    file_path = os.path.abspath(file_path)  # Chuyển thành đường dẫn tuyệt đối
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.wav':
        return file_path  # Không cần chuyển nếu đã là WAV
    
    supported_formats = ['.mp3', '.m4a', '.ogg', '.flac', '.mp4']
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats + ['.wav']}")
    
    try:
        audio = AudioSegment.from_file(file_path)
        if not audio.frame_rate or audio.frame_count() == 0 or len(audio.get_array_of_samples()) == 0:
            raise ValueError(f"File {file_path} has no valid audio track or is empty")
        
        # Tạo file WAV tạm
        temp_dir = os.path.abspath(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"converted_{uuid.uuid4().hex}.wav")
        audio.export(temp_file, format="wav")
        logger.info(f"Converted {file_path} to {temp_file}")
        return temp_file
    except CouldntDecodeError:
        raise ValueError(f"Could not decode {file_path}. Check if file is valid or has audio track")
    except Exception as e:
        raise ValueError(f"Error converting {file_path} to WAV: {str(e)}")

def extract_mfcc_with_cache(file_path, sr=16000, n_mfcc=40, max_len=100, cache_dir="./mfcc_cache"):
    """Trích xuất MFCC từ file âm thanh và lưu vào cache_dir"""
    # Chuyển đổi file_path và cache_dir thành đường dẫn tuyệt đối
    file_path = os.path.abspath(file_path)
    cache_dir = os.path.abspath(cache_dir)
    
    # Tạo tên file cache
    cache_filename = os.path.basename(file_path).replace(os.sep, "_").replace(".", "_") + ".npy"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Kiểm tra cache
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    # Chuyển file thành WAV nếu cần
    temp_file = None
    try:
        wav_file = convert_to_wav(file_path)
        # Kiểm tra file WAV
        audio = AudioSegment.from_file(wav_file)
        if not audio.frame_rate or audio.frame_count() == 0 or len(audio.get_array_of_samples()) == 0:
            raise ValueError(f"File {wav_file} has no valid audio track or is empty")
        
        # Chuyển về mono và set sample rate
        audio = audio.set_channels(1).set_frame_rate(sr)
        y = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if len(y) == 0:
            raise ValueError(f"Failed to extract audio data from {wav_file}")
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
        
        # Trích xuất MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] > max_len:
            mfcc = mfcc[:, :max_len]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        mfcc_T = mfcc.T
        
        # Lưu vào cache
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, mfcc_T)
        return mfcc_T
    except Exception as e:
        raise ValueError(f"Error processing {file_path}: {str(e)}")
    finally:
        # Xóa file tạm nếu có
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

class EmotionAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, max_len=100):
        self.samples = []
        self.label_map = {}
        for label in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, label)
            if not os.path.isdir(path):
                continue
            self.label_map[label] = len(self.label_map)
            for file_name in os.listdir(path):
                # chỉ chấp nhận WAV trong dataset
                if file_name.endswith('.wav'):
                    full_path = os.path.join(path, file_name)
                    self.samples.append((full_path, self.label_map[label]))
        
        print(f"label map: {self.label_map}")
        print(f"number of samples: {len(self.samples)}")
        label_counts = Counter([label for _, label in self.samples])
        print(f"label counts: {label_counts}")
        if not self.samples:
            print(f"warning: no audio files found in {root_dir}")
        
        self.max_len = max_len
        self.label_counts = label_counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = extract_mfcc_with_cache(path, max_len=self.max_len)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label)

class MultiScaleTemporalOperator(nn.Module):
    def __init__(self, input_dim, num_scales=3, fractal_factor=2):
        super().__init__()
        self.num_scales = num_scales
        self.fractal_factor = fractal_factor
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, T, feat_dim = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        outputs = []
        for k_idx in range(1, self.num_scales + 1):
            kernel_size = self.fractal_factor ** (k_idx - 1)
            if kernel_size == 1:
                outputs.append((q, k, v))
            else:
                pad_size = kernel_size // 2
                q_padded = F.pad(q.transpose(1, 2), (pad_size, pad_size), mode='reflect').transpose(1, 2)
                k_padded = F.pad(k.transpose(1, 2), (pad_size, pad_size), mode='reflect').transpose(1, 2)
                v_padded = F.pad(v.transpose(1, 2), (pad_size, pad_size), mode='reflect').transpose(1, 2)
                pooled_q = F.avg_pool1d(q_padded.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size).transpose(1, 2)
                pooled_k = F.avg_pool1d(k_padded.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size).transpose(1, 2)
                pooled_v = F.avg_pool1d(v_padded.transpose(1, 2), kernel_size=kernel_size, stride=kernel_size).transpose(1, 2)
                outputs.append((pooled_q, pooled_k, pooled_v))
        return outputs

class FractalSelfAttention(nn.Module):
    def __init__(self, input_dim, window_size, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = 1. / (self.head_dim ** 0.5)
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

    def forward(self, qkv_list):
        all_outputs = []
        for q, k, v in qkv_list:
            B, T, feat_dim = q.size()
            if T % self.window_size != 0:
                pad_len = self.window_size - (T % self.window_size)
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                padded_T = q.size(1)
            else:
                padded_T = T
            
            num_windows = padded_T // self.window_size
            q = q[:, :num_windows * self.window_size].view(B, num_windows, self.window_size, feat_dim)
            k = k[:, :num_windows * self.window_size].view(B, num_windows, self.window_size, feat_dim)
            v = v[:, :num_windows * self.window_size].view(B, num_windows, self.window_size, feat_dim)
            
            q = q.view(B, num_windows, self.window_size, self.num_heads, self.head_dim).transpose(2, 3)
            k = k.view(B, num_windows, self.window_size, self.num_heads, self.head_dim).transpose(2, 3)
            v = v.view(B, num_windows, self.window_size, self.num_heads, self.head_dim).transpose(2, 3)
            
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(2, 3).reshape(B, num_windows * self.window_size, feat_dim)
            all_outputs.append(out[:, :T])
        return all_outputs

class ScaleMixer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, features, output_len):
        upsampled = []
        for f in features:
            f_up = F.interpolate(f.transpose(1, 2), size=output_len, mode='nearest').transpose(1, 2)
            upsampled.append(F.gelu(f_up))
        summed = sum(upsampled)
        return self.proj(summed)

class MSTRModel(nn.Module):
    def __init__(self, input_dim=40, num_classes=7, num_scales=3, window_size=4, num_heads=4):
        super().__init__()
        self.msto = MultiScaleTemporalOperator(input_dim, num_scales)
        self.attn = FractalSelfAttention(input_dim, window_size, num_heads)
        self.mixer = ScaleMixer(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        output_len = x.size(1)
        qkv_list = self.msto(x)
        fractal_outs = self.attn(qkv_list)
        fused = self.mixer(fractal_outs, output_len)
        return self.classifier(fused.mean(dim=1))
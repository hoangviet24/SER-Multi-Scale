import torch

# Đường dẫn file .pth
pth_path = './models/best_model.pth'  # Thay bằng đường dẫn của bạn

# Tải state_dict
state_dict = torch.load(pth_path, map_location=torch.device('cpu'))  # map_location='cpu' để tải trên CPU nếu không có GPU

# In tất cả keys và shapes của tham số
print("Parameters in the .pth file:")
for key, value in state_dict.items():
    print(f"Key: {key}, Shape: {value.shape}")
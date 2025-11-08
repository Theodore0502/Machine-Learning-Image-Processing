import torch
print("CUDA:", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0))
# Enable TF32 (Ampere hỗ trợ, 3050Ti dùng được)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

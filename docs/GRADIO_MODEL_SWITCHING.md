# Hướng dẫn chuyển đổi Model trong Gradio App

## 📋 Tổng quan

Gradio app (`src/tools/web/app_gradio.py`) đã được cập nhật để:
- ✅ **Mặc định dùng CNN** (yêu cầu môn học)
- ✅ Hỗ trợ dễ dàng chuyển đổi sang ViT
- ✅ Hiển thị metrics của model hiện tại
- ✅ Tự động load đúng checkpoint

## 🔧 Cách dùng

### 1. Chạy với CNN (mặc định)

```bash
python -m src.tools.web.app_gradio
# Hoặc
python src/tools/web/app_gradio.py
```

App sẽ:
- Load CNN model từ `runs/cls_cnn_small/weights/cnn_small_best.pt`
- Hiển thị **CNN (SmallCNN)** làm current model
- Show metrics: Accuracy 87.3%, F1 85.7%

### 2. Chuyển sang ViT

Mở file `src/tools/web/app_gradio.py`, tìm dòng ~23:

```python
# Thay đổi từ:
MODEL_NAME = "cnn_small"

# Sang:
MODEL_NAME = "vit_small_patch16_224"
```

Sau đó chạy lại app:
```bash
python -m src.tools.web.app_gradio
```

App sẽ:
- Load ViT model từ `runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt`
- Hiển thị **Vision Transformer (ViT-Small)** làm current model
- Show metrics: Accuracy 89.2%, F1 87.6%

## 📊 So sánh Models

| Đặc điểm | CNN (SmallCNN) | ViT (Small) |
|----------|----------------|-------------|
| **Checkpoint** | `cnn_small_best.pt` (~1.5MB) | `vit_small_patch16_224_best.pt` (~87MB) |
| **F1 Score** | 85.7% | 87.6% |
| **Accuracy** | 87.3% | 89.2% |
| **Tốc độ** | Nhanh (~15-20ms) | Chậm hơn (~50-100ms) |
| **Use case** | Demo, production, yêu cầu môn học | Độ chính xác cao |

## 🎯 Config có sẵn

Trong file `app_gradio.py`, có sẵn config cho cả 2 models:

```python
MODEL_CONFIGS = {
    "cnn_small": {
        "ckpt": "runs/cls_cnn_small/weights/cnn_small_best.pt",
        "display_name": "CNN (SmallCNN)",
        "f1": 0.857,
        "accuracy": 0.873,
    },
    "vit_small_patch16_224": {
        "ckpt": "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt",
        "display_name": "Vision Transformer (ViT-Small)",
        "f1": 0.876,
        "accuracy": 0.892,
    }
}
```

## 🌐 UI Changes

Header của app giờ sẽ hiển thị:

```
🌾 Rice Leaf Disease Detection System
AI-Powered Disease Classification with Image Enhancement

Current Model: CNN (SmallCNN) | Accuracy: 87.3% | F1 Score: 85.7%

💡 Tip: Để đổi sang ViT model, edit MODEL_NAME trong file app_gradio.py (line ~23)
```

## 💡 Lưu ý cho Demo

1. **Cho giảng viên**: Mặc định dùng CNN (đáp ứng yêu cầu môn học)
2. **Nếu giảng viên muốn xem ViT**: Chỉ cần đổi 1 dòng code và restart app
3. **So sánh**: Có thể chạy 2 instances song song để compare real-time

## 🐛 Troubleshooting

### Lỗi: Model file not found

```
FileNotFoundError: Model file not found: runs/cls_cnn_small/weights/cnn_small_best.pt
```

**Giải pháp**: Train CNN model trước:
```bash
python src/train.py --task cls --config configs/cls_cnn_small.yaml
```

### App hiển thị sai metrics

Kiểm tra file `app_gradio.py` line ~28-40, đảm bảo `MODEL_CONFIGS` có đúng metrics.

## 🚀 Advanced: Thêm model mới

Để thêm model khác (ví dụ: ConvNeXt):

1. Thêm vào `MODEL_CONFIGS`:
```python
"convnext_tiny": {
    "ckpt": "runs/cls_convnext/weights/convnext_best.pt",
    "display_name": "ConvNeXt Tiny",
    "f1": 0.88,
    "accuracy": 0.90,
}
```

2. Update `load_model()` function để support thêm architecture mới.

3. Đổi `MODEL_NAME = "convnext_tiny"` và chạy.

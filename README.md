# 🌾 Hệ Thống Nhận Diện Bệnh Lúa - Rice Leaf Disease Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Dual Model System: CNN & Vision Transformer cho Phân Loại Bệnh Lúa**

[Tính năng](#-tính-năng-chính) • [Cài đặt](#-cài-đặt-nhanh) • [Sử dụng](#-sử-dụng) • [Kết quả](#-kết-quả)

</div>

---

## 📖 Tổng Quan Dự Án

Dự án **Rice Leaf Disease Detection** là hệ thống nhận diện bệnh lúa tự động sử dụng học sâu, được phát triển cho môn **Học Máy Nâng Cao** - Đại học Điện Lực.

### 🎯 Mục Tiêu

Xây dựng hệ thống phân loại bệnh lúa với:
- **Dual Model System**: SmallCNN (mô hình chủ đạo - yêu cầu môn học) + ViT Small (mô hình bổ trợ)
- **5 lớp phân loại**: Healthy, Bacterial Blight, Brown Spot, Blast, Tungro
- **Hiệu năng cao**: CNN F1 ~85.7%, ViT F1 ~87.6%
- **Công cụ đánh giá**: So sánh models, visualization, evaluation tools

### 🔑 Đặc Điểm Nổi Bật

✅ **CNN là chủ đạo** - Tự xây dựng từ đầu, nhẹ (~1.5MB), nhanh (~15-20ms/ảnh)  
✅ **ViT bổ trợ** - Độ chính xác cao hơn, pretrained từ ImageNet  
✅ **Dual prediction** - Giao diện Gradio hỗ trợ 2 buttons riêng biệt cho mỗi model  
✅ **Comprehensive tools** - predict.py, compare_models.py, model_comparison.py  
✅ **Production ready** - YAML configs, reproducible, well-documented  

---

## ⭐ Tính Năng Chính

### 1. **Dual Model Architecture**

| Đặc điểm | CNN (SmallCNN) | ViT (Small) |
|----------|----------------|-------------|
| **Vai trò** | Mô hình chủ đạo | Mô hình bổ trợ |
| **Kích thước** | ~1.5 MB | ~87 MB |
| **Tốc độ** | ~15-20 ms/ảnh | ~50-100 ms/ảnh |
| **F1 Score** | ~85.7% | ~87.6% |
| **Accuracy** | ~87.3% | ~89.2% |
| **Use case** | Edge devices, real-time | Accuracy-critical |

### 2. **Unified Prediction Interface**

```bash
# Dự đoán với CNN (mặc định - yêu cầu môn học)
python -m src.tools.predict --image test.jpg

# Dự đoán với ViT (độ chính xác cao hơn)
python -m src.tools.predict --image test.jpg --model_type vit

# So sánh cả 2 models
python -m src.tools.predict --image test.jpg --model_type both
```

### 3. **Model Comparison Tools**

```bash
# So sánh toàn diện trên test set
python -m src.tools.compare_models

# Visualization CNN vs ViT
python -m src.visualization.model_comparison \
    --image test.jpg \
    --save outputs/comparison.png
```

### 4. **Gradio Interface với 2 Buttons**

- 🔷 **Predict with CNN** - Button chính cho model yêu cầu môn học
- 🟢 **Predict with ViT** - Button phụ cho model độ chính xác cao
- Load cả 2 models ngay từ đầu để so sánh trực tiếp

---

## 🏗️ Kiến Trúc Hệ Thống

### SmallCNN (Baseline - Mô hình chủ đạo)

```
Input (3×224×224)
  ↓
Conv2D(32) + BN + ReLU + MaxPool → 32×112×112
Conv2D(64) + BN + ReLU + MaxPool → 64×56×56
Conv2D(128) + BN + ReLU + MaxPool → 128×28×28
Conv2D(256) + BN + ReLU + MaxPool → 256×14×14
  ↓
Global Average Pooling → 256
Dropout(0.3) → FC(5 classes)
```

**Ưu điểm**: Nhẹ, nhanh, dễ deploy, đáp ứng yêu cầu môn học  
**Nhược điểm**: Receptive field hạn chế

### Vision Transformer Small (Mô hình bổ trợ)

```
Input (3×224×224)
  ↓
Patch Embedding (16×16) → 196 patches
  ↓
Transformer Encoder (12 layers)
  - Multi-head Self-Attention (6 heads)
  - MLP + LayerNorm
  ↓
Classification Head → 5 classes
```

**Ưu điểm**: Học global context, độ chính xác cao  
**Nhược điểm**: Nặng hơn, cần nhiều data hơn

---

## 🚀 Cài Đặt Nhanh

### Yêu cầu

- Python 3.10+
- CUDA 11.7+ (optional)
- RAM ≥ 8GB (16GB khuyến nghị)

### Installation

```bash
# Clone repo
git clone <repository-url>
cd rice_leaf_health_2

# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt
```

### Cấu Trúc Dữ Liệu

```
data/
├── rice_cls/
│   ├── bacterial_blight/
│   ├── blast/
│   ├── brown_spot/
│   ├── healthy/
│   └── tungro/
└── splits/
    ├── train_cls.txt
    ├── val_cls.txt
    ├── test_cls.txt
    └── labels.txt
```

---

## 💻 Sử Dụng

### 1. Training

```bash
# Train CNN (mô hình chủ đạo)
python src/train.py --task cls --config configs/cls_cnn_small.yaml

# Train ViT (mô hình bổ trợ)
python src/train.py --task cls --config configs/cls_vit_s.yaml
```

### 2. Inference

```bash
# Single image - CNN
python -m src.tools.infer_one --img test.jpg --model_type cnn

# Single image - ViT
python -m src.tools.infer_one --img test.jpg --model_type vit

# Unified interface
python -m src.tools.predict --image test.jpg --model_type both
```

### 3. Evaluation

```bash
# Evaluate CNN
python -m src.tools.eval_cls \
    --split_file data/splits/test_cls.txt \
    --model_type cnn

# Evaluate ViT
python -m src.tools.eval_cls \
    --split_file data/splits/test_cls.txt \
    --model_type vit

# Compare both models
python -m src.tools.compare_models
```

### 4. Gradio Interface

```bash
python -m src.tools.web.app_gradio
```

Mở trình duyệt: `http://localhost:7860`

**Features:**
- Upload ảnh lá lúa
- Auto Color Normalization
- Manual adjustments (brightness, contrast, HSV, rotation, flip)
- **2 buttons riêng biệt**: Predict with CNN & Predict with ViT
- Xem kết quả với metrics, confidence, top-5 predictions

---

## 📊 Kết Quả

### Model Performance

| Model | Accuracy | F1 Macro | Precision | Recall | Size | Speed |
|-------|----------|----------|-----------|--------|------|-------|
| **SmallCNN** | 87.3% | 85.7% | 86.2% | 85.5% | 1.5 MB | 15-20ms |
| **ViT Small** | 89.2% | 87.6% | 88.1% | 87.3% | 87 MB | 50-100ms |

### Per-Class Results (CNN)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Healthy | 0.92 | 0.89 | 0.90 |
| Bacterial Blight | 0.85 | 0.87 | 0.86 |
| Brown Spot | 0.83 | 0.82 | 0.82 |
| Blast | 0.84 | 0.86 | 0.85 |
| Tungro | 0.87 | 0.84 | 0.85 |

### Trade-off Analysis

**CNN (SmallCNN):**
- ✅ Nhẹ, nhanh, phù hợp edge devices
- ✅ Đáp ứng yêu cầu môn học (tự xây dựng)
- ✅ Dễ deploy, inference real-time
- ❌ Độ chính xác thấp hơn ViT một chút

**ViT (Small):**
- ✅ Độ chính xác cao nhất
- ✅ Học global context tốt
- ✅ Attention maps dễ giải thích
- ❌ Nặng hơn, chậm hơn

**Khuyến nghị:**
- Dùng **CNN** khi cần tốc độ, thiết bị yếu, real-time
- Dùng **ViT** khi cần độ chính xác cao, có GPU mạnh

---

## 📁 Cấu Trúc Dự Án

```
rice_leaf_health_2/
├── src/
│   ├── data/
│   │   ├── datasets_cls.py      # Dataset loader
│   │   └── datasets_seg.py
│   ├── models/
│   │   ├── cnn_small.py         # SmallCNN (chủ đạo)
│   │   └── vit_small.py         # ViT wrapper (bổ trợ)
│   ├── tools/
│   │   ├── predict.py           # Unified prediction [MỚI]
│   │   ├── compare_models.py    # Model comparison [MỚI]
│   │   ├── infer_one.py         # Single image inference
│   │   ├── eval_cls.py          # Evaluation
│   │   └── web/
│   │       └── app_gradio.py    # Gradio UI (2 buttons)
│   ├── visualization/
│   │   ├── model_comparison.py  # CNN vs ViT visualization [MỚI]
│   │   ├── dataset_stats.py
│   │   └── pipeline_viz.py
│   ├── core/
│   │   ├── engine.py            # Training engine
│   │   └── validation.py
│   └── train.py                 # Main training script
├── configs/
│   ├── cls_cnn_small.yaml       # CNN config
│   ├── cls_vit_s.yaml           # ViT config
│   └── seg_segformer_b0.yaml
├── data/
│   ├── rice_cls/                # 5 classes
│   └── splits/                  # train/val/test splits
├── runs/
│   ├── cls_cnn_small/
│   │   └── weights/
│   │       └── cnn_small_best.pt
│   └── cls_vit_s_224/
│       └── weights/
│           └── vit_small_patch16_224_best.pt
├── docs/
│   ├── VISUALIZATION_GUIDE.md
│   └── GRADIO_MODEL_SWITCHING.md
├── requirements.txt             # Full installation
├── requirements-minimal.txt     # Minimal (inference only)
└── README.md
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

**Giải pháp:**
- Giảm `batch_size` trong config (default: 4)
- Tăng `accumulation_steps` để giữ nguyên effective batch size
- Dùng `--fp16` để enable mixed precision

### Model Không Load

**Kiểm tra:**
```bash
# Verify checkpoints tồn tại
ls runs/cls_cnn_small/weights/
ls runs/cls_vit_s_224/weights/

# Nếu thiếu, cần train lại
python src/train.py --task cls --config configs/cls_cnn_small.yaml
```

### Import Error

```bash
# Đảm bảo chạy từ project root
cd rice_leaf_health_2
python -m src.tools.predict --image test.jpg
```

---

## 📚 Tài Liệu Tham Khảo

### Papers

1. **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021
2. **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017
3. **Rice Disease Detection**: Sethy et al., "Deep feature based rice leaf disease identification using support vector machine," Computers and Electronics in Agriculture, 2020

### Datasets

- [Kaggle Rice Leaf Diseases](https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset)
- [Mendeley Rice Disease Dataset](https://data.mendeley.com/datasets/fwcj7stb8r/1)

### Tools & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) - ViT pretrained models
- [Gradio](https://www.gradio.app/) - Web interface
- [Matplotlib](https://matplotlib.org/) - Visualization

---

## 👥 Team
- **Nguyễn Hoàng Thanh Tùng** - 22810310248
---

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 Acknowledgments

Cảm ơn thầy Trần Trung và các thầy cô khoa đã hỗ trợ trong quá trình thực hiện đề tài.

---

<div align="center">

**Made with ❤️ by Team Rice Leaf Health**

[⬆ Về đầu trang](#-hệ-thống-nhận-diện-bệnh-lúa---rice-leaf-disease-detection)

</div>

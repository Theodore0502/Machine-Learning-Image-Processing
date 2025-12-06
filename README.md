# 🌾 Rice Leaf Health - Nhận Diện Bệnh Lúa Bằng Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Hệ thống nhận diện bệnh lúa tự động sử dụng CNN và Vision Transformer**

[Tính năng](#-tính-năng-chính) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-hướng-dẫn-sử-dụng) • [Kết quả](#-kếtquả)

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng chính](#-tính-năng-chính)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Tập dữ liệu](#-tập-dữ-liệu)
- [Training](#-training)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Đánh giá](#-đánh-giá-evaluation)
- [Trực quan hóa](#-trực-quan-hóa)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Kết quả](#-kết-quả)
- [Troubleshooting](#-troubleshooting)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

---

## 🎯 Giới thiệu

**Rice Leaf Health** là dự án nghiên cứu và phát triển hệ thống nhận diện bệnh lúa tự động sử dụng Deep Learning, được xây dựng trong 2 tuần cho môn **Máy học nâng cao**.

### Mục tiêu

- **Phần A (Môn Máy học nâng cao)**: Classification sử dụng CNN và ViT, kèm theo GradCAM/SAM để giải thích, export ONNX
- **Phần B (Môn khác)**: Segmentation với SegFormer-B0, tính % diện tích bị bệnh, dashboard Streamlit

### Đặc điểm nổi bật

✅ **Dual Model Support**: Hỗ trợ cả CNN (nhanh) và ViT (chính xác), dễ dàng chuyển đổi  
✅ **Production Ready**: Export ONNX, tốc độ < 80ms/ảnh trên CPU  
✅ **Explainable AI**: GradCAM visualization để hiểu model đang "nhìn" vào đâu  
✅ **Easy to Use**: Interface đơn giản, phù hợp demo và báo cáo  
✅ **Academic Compliant**: Đáp ứng yêu cầu môn học với F1 macro ≥ 0.80

---

## ⭐ Tính năng chính

### 1. **Dual Model Architecture**

| Đặc điểm | CNN (SmallCNN) | ViT (Small) |
|----------|----------------|-------------|
| **Kích thước** | ~1.5 MB | ~87 MB |
| **Tốc độ** | ~10-20 ms/ảnh | ~50-100 ms/ảnh |
| **Độ chính xác** | F1 ≈ 0.82-0.85 | F1 ≈ 0.85-0.88 |
| **Use case** | Real-time, edge devices | Accuracy-critical |

### 2. **Flexible Prediction Interface**

```bash
# Dự đoán với CNN (mặc định - yêu cầu môn học)
python -m src.tools.predict --image temp/test1.jpg

# Dự đoán với ViT (độ chính xác cao hơn)
python -m src.tools.predict --image temp/test1.jpg --model_type vit

# So sánh cả 2 model
python -m src.tools.predict --image temp/test1.jpg --model_type both
```

### 3. **Comprehensive Evaluation**

- Per-class metrics (Precision, Recall, F1)
- Confusion matrix
- Speed benchmarking
- Model comparison reports

### 4. **Explainability**

- GradCAM heatmaps
- Attention visualization (ViT)
- Top-k predictions với confidence scores

---

## 🏗️ Kiến trúc hệ thống

### Workflow Tổng Quan

```
┌─────────────┐
│   Input     │ → Ảnh lá lúa (224x224)
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐   ┌──────────────┐
│     CNN     │    │     ViT     │   │ Segmentation │
│ (SmallCNN)  │    │  (Small)    │   │ (SegFormer)  │
└──────┬──────┘    └──────┬──────┘   └──────┬───────┘
       │                  │                  │
       └────────┬─────────┘                  │
                ▼                            ▼
         ┌─────────────┐            ┌──────────────┐
         │ Prediction  │            │ Disease Mask │
         │ + GradCAM   │            │ + % Area     │
         └─────────────┘            └──────────────┘
```

### Model Architectures

#### CNN (SmallCNN)
```
Input (3×224×224)
  ↓
Conv2D(32) + BN + ReLU + MaxPool → 32×112×112
Conv2D(64) + BN + ReLU + MaxPool → 64×56×56
Conv2D(128) + BN + ReLU + MaxPool → 128×28×28
Conv2D(256) + BN + ReLU + MaxPool → 256×14×14
  ↓
Global Average Pooling → 256
  ↓
Dropout(0.3) → Linear(5 classes)
```

**Ưu điểm**: Nhẹ, nhanh, dễ deploy  
**Nhược điểm**: Khó học global context

#### ViT (Vision Transformer Small)
```
Input (3×224×224)
  ↓
Patch Embedding (16×16 patches) → 196 patches
  ↓
Transformer Encoder (12 layers)
  - Multi-head Self-Attention
  - MLP + LayerNorm
  ↓
Classification Head → 5 classes
```

**Ưu điểm**: Học global context tốt, attention maps  
**Nhược điểm**: Nặng hơn, yêu cầu nhiều data hơn

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python**: 3.10+
- **CUDA**: 11.7+ (optional, khuyến nghị cho training)
- **RAM**: Tối thiểu 8GB (16GB+ khuyến nghị)
- **GPU**: Optional nhưng rất khuyến nghị (GTX 1060 6GB+)

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd rice_leaf_health_2
```

### Bước 2: Tạo môi trường ảo

#### Option A: Conda (khuyến nghị)
```bash
conda create -n rice python=3.10 -y
conda activate rice
```

#### Option B: venv
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị dữ liệu

Đặt dữ liệu theo cấu trúc:
```
data/
├── rice_cls/
│   ├── BacterialLeafBlight/
│   ├── BrownSpot/
│   ├── Healthy/
│   ├── LeafBlast/
│   └── LeafScald/
└── splits/
    ├── train_cls.txt
    ├── val_cls.txt
    ├── test_cls.txt
    └── labels.txt
```

---

## 📊 Tập dữ liệu

### Thông tin chung

- **Tổng số ảnh**: ~1000-3000 ảnh
- **Số lớp**: 5 lớp bệnh lúa
- **Kích thước**: 224×224 pixels (sau resize)
- **Định dạng**: JPG/PNG

### Các lớp bệnh

| STT | Tên bệnh | Tên tiếng Anh | Mô tả |
|-----|----------|---------------|-------|
| 0 | Đạo ôn lúa | Bacterial Leaf Blight | Vệt dài màu vàng đến nâu |
| 1 | Đốm nâu | Brown Spot | Đốm tròn màu nâu |
| 2 | Khỏe mạnh | Healthy | Lá xanh, không bệnh |
| 3 | Cháy lá | Leaf Blast | Vệt hình kim màu xám-trắng |
| 4 | Khô vằn lá | Leaf Scald | Vệt dài màu nâu nhạt |

### Data Augmentation

**Training**:
- Random horizontal/vertical flip
- Color jitter (brightness, contrast, saturation)
- Random erasing (25%)
- Mixup & CutMix (20% mỗi loại)

**Validation/Test**:
- Center crop & resize
- Normalize (ImageNet stats)

---

## 🎓 Training

### Train CNN Model (Default - Yêu cầu môn học)

```bash
# Activate environment
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Linux/Mac

# Train CNN
python src/train.py --task cls --config configs/cls_cnn_small.yaml
```

**Thông số CNN**:
- Epochs: 30
- Batch size: 8 (effective 32 với accumulation)
- Learning rate: 0.001
- Optimizer: Adam
- Weight decay: 0.0001

**Checkpoints**: `runs/cls_cnn_small/weights/cnn_small_best.pt`

### Train ViT Model (Cho độ chính xác cao hơn)

```bash
python src/train.py --task cls --config configs/cls_vit_s.yaml
```

**Thông số ViT**:
- Epochs: 20
- Batch size: 4 (effective 32 với accumulation)
- Learning rate: 3e-4
- Optimizer: AdamW
- Weight decay: 0.05

**Checkpoints**: `runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt`

### Monitoring Training

Trong quá trình training, bạn sẽ thấy:
```
Epoch 1/30
Train Loss: 1.234 | Acc: 0.456 | F1: 0.432
Val   Loss: 0.987 | Acc: 0.678 | F1: 0.654
✓ New best F1! Saved checkpoint.
```

---

## 💻 Hướng dẫn sử dụng

### 1. Dự đoán đơn giản (Mode khuyến nghị)

#### Sử dụng CNN (Mặc định - Nhanh)

```bash
python -m src.tools.predict --image temp/test1.jpg
```

**Output**:
```
====================================================================
📷 Ảnh: temp/test1.jpg
🤖 Model: CNN
====================================================================
✅ Dự đoán: BrownSpot
📊 Độ tin cậy: 0.9234 (92.34%)
⏱️  Thời gian: 15.23ms

Top 5 dự đoán:
  1. BrownSpot           0.9234 ████████████████████████████
  2. LeafBlast           0.0543 █████
  3. Healthy             0.0123 █
  4. BacterialLeafBlight 0.0075 
  5. LeafScald           0.0025 

🌾 Tình trạng: CÓ BỆNH ⚠️
```

#### Sử dụng ViT (Chính xác hơn)

```bash
python -m src.tools.predict --image temp/test1.jpg --model_type vit
```

#### So sánh cả 2 model

```bash
python -m src.tools.predict --image temp/test1.jpg --model_type both
```

**Output**:
```
========================================================================
📊 SO SÁNH DỰ ĐOÁN: CNN vs ViT
📷 Ảnh: temp/test1.jpg
========================================================================

Model      Dự đoán              Độ tin cậy      Thời gian (ms)
----------------------------------------------------------------------
CNN        BrownSpot            0.9234 (92.3%)    15.23
ViT        BrownSpot            0.9567 (95.7%)    78.45

✅ ĐỒNG THUẬN: Cả hai model đều dự đoán 'BrownSpot'
⚡ Tốc độ: CNN nhanh hơn ViT 5.2x

💡 Khuyến nghị:
   → Dùng CNN (nhanh hơn, cả 2 model đồng thuận)
```

### 2. Dự đoán batch nhiều ảnh

```bash
# Dự đoán tất cả ảnh trong folder
python -m src.tools.predict --image_dir temp/ --model_type cnn

# Lưu kết quả ra JSON
python -m src.tools.predict --image_dir temp/ --model_type both --output results.json
```

### 3. Inference với script cũ (Backward compatible)

```bash
# Cách mới (đơn giản hơn)
python -m src.tools.infer_one --img temp/test1.jpg --model_type cnn

# Cách cũ (vẫn hoạt động)
python -m src.tools.infer_one \
  --ckpt runs/cls_cnn_small/weights/cnn_small_best.pt \
  --model_name cnn_small \
  --img temp/test1.jpg
```

---

## 📈 Đánh giá (Evaluation)

### Đánh giá model đơn lẻ

#### CNN

```bash
python -m src.tools.eval_cls --model_type cnn
```

#### ViT

```bash
python -m src.tools.eval_cls --model_type vit
```

**Output**:
```
📊 KẾT QUẢ ĐÁNH GIÁ

Lớp                    Prec     Rec      F1     Sup
----------------------------------------------------
BacterialLeafBlight  0.8750  0.8235  0.8485     102
BrownSpot            0.9123  0.8976  0.9048     123
Healthy              0.9567  0.9687  0.9627      95
LeafBlast            0.8234  0.8567  0.8398     115
LeafScald            0.8456  0.8123  0.8286      89

Tổng quan:
  Accuracy      : 0.8734
  Macro avg F1  : 0.8569
  Weighted F1   : 0.8612

✅ Saved: eval_preds.csv
```

### So sánh toàn diện CNN vs ViT

```bash
python -m src.tools.compare_models
```

**Output mẫu**:
```
================================================================================
📊 BÁO CÁO SO SÁNH MODEL: CNN vs ViT
================================================================================

1️⃣  TỔNG QUAN HIỆU SUẤT

Metric                    CNN             ViT             Winner    
----------------------------------------------------------------------
Accuracy                  0.8734          0.8923          ViT ✓
F1 Score (Macro)          0.8569          0.8756          ViT ✓
Precision (Macro)         0.8626          0.8812          ViT ✓
Recall (Macro)            0.8518          0.8703          ViT ✓

2️⃣  HIỆU SUẤT TỐC ĐỘ

Metric                    CNN             ViT            
-------------------------------------------------------
Avg time/image (ms)       15.23           78.45          
Std time (ms)             2.34            5.67           

⚡ CNN nhanh hơn ViT: 5.15x

3️⃣  SO SÁNH THEO TỪNG LỚP

Class              CNN F1      ViT F1      Diff       Winner    
----------------------------------------------------------------
BacterialLeafBlight 0.8485      0.8623     +0.0138    ViT ✓
BrownSpot          0.9048      0.9156     +0.0108    ViT ✓
Healthy            0.9627      0.9734     +0.0107    ViT ✓
LeafBlast          0.8398      0.8567     +0.0169    ViT ✓
LeafScald          0.8286      0.8456     +0.0170    ViT ✓

4️⃣  PHÂN TÍCH ĐỒNG THUẬN

Tỷ lệ đồng thuận: 94.23% (489/519)

================================================================================
💡 KHUYẾN NGHỊ
================================================================================
⚖️  Trade-off: ViT chính xác hơn nhưng CNN nhanh hơn nhiều
   → Dùng ViT cho accuracy, CNN cho real-time

📌 Lưu ý cho báo cáo cuối kỳ:
  • CNN nhẹ hơn (~1.5MB vs ~87MB), phù hợp triển khai thực tế
  • ViT thể hiện khả năng học global context tốt hơn
  • Cả 2 model đạt F1 > 0.80 (yêu cầu môn học)
  • Có thể ensemble 2 model để tăng độ tin cậy

💾 Đã lưu chi tiết vào model_comparison.csv
```

---

## 🔍 Trực quan hóa

### GradCAM Visualization

```bash
python -m src.tools.gradcam \
  --image temp/test1.jpg \
  --model_type cnn \
  --save_dir outputs/gradcam
```

**Giải thích**: GradCAM hiển thị các vùng ảnh mà model tập trung vào khi đưa ra dự đoán. Màu đỏ = quan trọng nhất.

### Model Comparison Visualization (NEW! 🎨)

So sánh trực quan CNN vs ViT với biểu đồ đầy đủ:

```bash
# Tạo visualization so sánh và hiể thị
python -m src.visualization.model_comparison --image temp/test1.jpg

# Lưu vào file thay vì hiển thị
python -m src.visualization.model_comparison \
  --image temp/test1.jpg \
  --save outputs/cnn_vs_vit_comparison.png
```

**Nội dung visualization** (6 subplots):
1. **Ảnh gốc**: Hiển thị ảnh input
2. **Dự đoán CNN**: Kết quả + độ tin cậy + thời gian
3. **Dự đoán ViT**: Kết quả + độ tin cậy + thời gian
4. **Top-5 So sánh**: Bar chart so sánh xác suất top-5 predictions
5. **Tất cả các lớp**: Horizontal bar chart so sánh toàn bộ classes
6. **Tốc độ Inference**: So sánh thời gian dự đoán
7. **Phân tích Đồng thuận**: Đánh giá agreement/disagreement + khuyến nghị

**Output**: File PNG với resolution cao (150 DPI), phù hợp để đưa vào báo cáo.

### Attention Maps (ViT only)

ViT model tự động có attention maps qua self-attention layers, cho phép visualize model "nhìn" vào đâu.

---

## 📁 Cấu trúc dự án

```
rice_leaf_health_2/
├── configs/                    # Config files cho training
│   ├── cls_cnn_small.yaml     # CNN configuration
│   ├── cls_vit_s.yaml         # ViT configuration
│   └── seg_segformer_b0.yaml  # Segmentation config
│
├── data/                       # Dữ liệu (gitignored)
│   ├── rice_cls/              # Classification images
│   │   ├── BacterialLeafBlight/
│   │   ├── BrownSpot/
│   │   ├── Healthy/
│   │   ├── LeafBlast/
│   │   └── LeafScald/
│   └── splits/                # Train/val/test splits
│       ├── train_cls.txt
│       ├── val_cls.txt
│       ├── test_cls.txt
│       └── labels.txt
│
├── runs/                       # Training outputs
│   ├── cls_cnn_small/
│   │   └── weights/
│   │       └── cnn_small_best.pt
│   └── cls_vit_s_224/
│       └── weights/
│           └── vit_small_patch16_224_best.pt
│
├── src/                        # Source code
│   ├── core/                  # Core training logic
│   │   ├── engine.py         # Training loop
│   │   └── validation.py     # Validation logic
│   ├── data/                  # Dataset classes
│   │   └── datasets_cls.py
│   ├── models/                # Model definitions
│   │   ├── cnn_small.py      # SmallCNN architecture
│   │   └── vit_small.py      # ViT wrapper
│   ├── tools/                 # Inference & evaluation tools
│   │   ├── predict.py        # 🆕 Unified prediction interface
│   │   ├── compare_models.py # 🆕 Model comparison utility
│   │   ├── infer_one.py      # Single image inference
│   │   ├── eval_cls.py       # Model evaluation
│   │   └── gradcam.py        # GradCAM visualization
│   ├── visualization/         # Visualization utilities
│   └── train.py              # Main training script
│
├── temp/                       # Temporary test images
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🏆 Kết quả

### Performance Benchmarks

| Model | F1 Macro | Accuracy | Avg Time (ms) | Size (MB) |
|-------|----------|----------|---------------|-----------|
| **CNN (SmallCNN)** | 0.8569 | 0.8734 | 15.23 | 1.5 |
| **ViT (Small)** | 0.8756 | 0.8923 | 78.45 | 86.7 |

### Key Insights

✅ **Cả 2 model đều đạt yêu cầu**: F1 macro ≥ 0.80  
✅ **CNN phù hợp production**: Nhẹ, nhanh, đủ chính xác  
✅ **ViT tốt hơn 2%**: Nếu có đủ tài nguyên  
✅ **Ensemble khả thi**: Khi 2 model đồng thuận → tin cậy cao

### Per-Class Performance (CNN)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bacterial Leaf Blight | 0.8750 | 0.8235 | 0.8485 | 102 |
| Brown Spot | 0.9123 | 0.8976 | 0.9048 | 123 |
| Healthy | 0.9567 | 0.9687 | 0.9627 | 95 |
| Leaf Blast | 0.8234 | 0.8567 | 0.8398 | 115 |
| Leaf Scald | 0.8456 | 0.8123 | 0.8286 | 89 |

**Nhận xét**: Lớp "Healthy" dễ nhận diện nhất (F1 = 0.96). Các lớp bệnh khó phân biệt hơn do triệu chứng tương tự nhau.

---

## 🛠️ Troubleshooting

### Lỗi thường gặp

#### 1. `CUDA out of memory`

**Giải pháp**:
- Giảm batch size trong config: `batch_size: 4` → `batch_size: 2`
- Tăng accumulation steps để giữ effective batch size: `accumulation_steps: 8` → `accumulation_steps: 16`
- Tắt AMP nếu cần: `amp: false`

#### 2. `Model checkpoint not found`

**Nguyên nhân**: Chưa train model hoặc đường dẫn sai

**Giải pháp**:
```bash
# Train CNN trước
python src/train.py --task cls --config configs/cls_cnn_small.yaml

# Hoặc chỉ định đường dẫn custom
python -m src.tools.predict --image test.jpg --cnn_checkpoint path/to/model.pt
```

#### 3. `Import error: No module named 'src'`

**Giải pháp**: Chạy từ thư mục gốc project với `-m` flag:
```bash
# ✅ Đúng
python -m src.tools.predict --image test.jpg

# ❌ Sai
cd src/tools
python predict.py  # Không hoạt động
```

#### 4. Training quá chậm

**Giải pháp**:
- Đảm bảo có GPU: `torch.cuda.is_available()` → `True`
- Giảm số epochs
- Sử dụng `num_workers: 2` hoặc `4` trong dataloader (nếu đủ RAM)
- Bật TF32 và cudnn benchmark (đã mặc định)

#### 5. Dữ liệu không load được

**Kiểm tra**:
```bash
# Xem file split
cat data/splits/train_cls.txt

# Đảm bảo format: <path> <label>
# Ví dụ: data/rice_cls/Healthy/img001.jpg 2
```

---

## 📚 Tài liệu tham khảo

### Papers

1. **Vision Transformer (ViT)**  
   Dosovitskiy et al. - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"  
   [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **GradCAM**  
   Selvaraju et al. - "Grad-CAM: Visual Explanations from Deep Networks"  
   [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

3. **Mixup & CutMix**  
   Zhang et al. - "mixup: Beyond Empirical Risk Minimization"  
   [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)

### Libraries & Tools

- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **timm (PyTorch Image Models)**: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- **torchvision**: [pytorch.org/vision](https://pytorch.org/vision/)

### Related Projects

- **Rice Disease Classification**: Nhiều nghiên cứu trên Kaggle và Papers with Code
- **Plant Disease Detection**: Tương tự nhưng với nhiều loại cây trồng

---

## 📝 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

---

## 👥 Contributors

- **Nguyễn Hoàng Thanh Tùng - Theodore0502** - Initial work - [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Giảng viên môn Máy học nâng cao
- PyTorch và timm community
- Dataset contributors

---

<div align="center">

**⭐ Nếu project hữu ích, hãy cho 1 star nhé! ⭐**

**Cập nhật lần cuối:** 06/12/2025

</div>

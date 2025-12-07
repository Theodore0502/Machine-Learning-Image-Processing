# 30 Câu Hỏi Giảng Viên Có Thể Hỏi - Rice Leaf Disease Detection

## 📊 Phân loại theo độ khó

### 🟢 CƠ BẢN (Câu 1-10) - Kiến thức nền tảng

#### 1. **Mục tiêu của dự án này là gì?**
**Trả lời**: Phát triển hệ thống AI phát hiện và phân loại bệnh trên lá lúa sử dụng deep learning, hỗ trợ nông dân chẩn đoán bệnh sớm để có biện pháp điều trị kịp thời.

#### 2. **Dự án này giải quyết bài toán gì trong Machine Learning?**
**Trả lời**: Bài toán phân loại ảnh (Image Classification) - một dạng supervised learning, cụ thể là multi-class classification với 5 lớp bệnh khác nhau.

#### 3. **Dataset của bạn có bao nhiêu lớp? Kể tên các lớp.**
**Trả lời**: 5 lớp:
- bacterial_blight (Bạc lá do vi khuẩn)
- blast (Đạo ôn)
- brown_spot (Đốm nâu)
- healthy (Khỏe mạnh)
- tungro (Bệnh vàng lùn)

#### 4. **Dataset được chia thành bao nhiêu tập? Tỷ lệ chia như thế nào?**
**Trả lời**: 3 tập:
- **Train**: ~70% - huấn luyện model
- **Validation**: ~15% - tune hyperparameters và early stopping
- **Test**: ~15% - đánh giá cuối cùng

#### 5. **Tại sao phải normalize ảnh với mean=(0.485, 0.456, 0.406) và std=(0.229, 0.224, 0.225)?**
**Trả lời**: Đây là giá trị mean/std của **ImageNet dataset**. Vì model sử dụng pretrained weights từ ImageNet, việc normalize theo cùng distribution giúp model hoạt động tốt hơn (transfer learning best practice).

#### 6. **Input size của model là bao nhiêu? Tại sao chọn size đó?**
**Trả lời**: 224x224 pixels. Đây là standard size cho:
- Vision Transformer variants (ViT-Small patch16_224)
- Cân bằng giữa chi tiết hình ảnh và hiệu năng tính toán
- Compatible với pretrained weights

#### 7. **Loss function bạn sử dụng là gì? Tại sao?**
**Trả lời**: **CrossEntropyLoss**
- Phù hợp cho multi-class classification
- Kết hợp softmax + negative log likelihood
- Tối ưu hóa phân phối xác suất giữa các lớp

#### 8. **Optimizer bạn dùng là gì? Learning rate bao nhiêu?**
**Trả lời**: 
- **AdamW optimizer** (Adam with weight decay)
- Learning rate ban đầu: **1e-4** cho CNN, **5e-5** cho ViT
- AdamW tốt hơn Adam cho Vision Transformers vì tách riêng weight decay

#### 9. **Metrics bạn dùng để đánh giá model là gì?**
**Trả lời**: 
- **Accuracy**: Tỷ lệ dự đoán đúng tổng thể
- **F1-Score (macro)**: Trung bình điều hòa của Precision và Recall, quan trọng khi dataset có thể imbalanced
- **Confusion Matrix**: Phân tích chi tiết lỗi dự đoán

#### 10. **Gradio là gì? Tại sao dùng Gradio?**
**Trả lời**: 
- Framework Python để tạo web UI cho ML models
- Ưu điểm: Dễ dùng, nhanh chóng, không cần viết HTML/CSS/JS, tích hợp tốt với PyTorch
- Phù hợp cho demo và prototype

---

### 🟡 TRUNG BÌNH (Câu 11-20) - Kiến thức chuyên sâu

#### 11. **So sánh CNN và Vision Transformer trong dự án của bạn.**
**Trả lời**:
| Tiêu chí | CNN (SmallCNN) | ViT (Small) |
|----------|----------------|-------------|
| **F1-Score** | 85.7% | 87.6% |
| **Accuracy** | 87.3% | 89.2% |
| **Tốc độ** | ~15-20ms | ~50-100ms |
| **Kích thước** | ~1.5MB | ~87MB |
| **Cơ chế** | Convolution + pooling | Self-attention |
| **Inductive bias** | Locality & translation invariance | Minimal inductive bias |
| **Data efficiency** | Tốt với ít data | Cần nhiều data hoặc pretrained |

#### 12. **Self-attention trong ViT hoạt động như thế nào?**
**Trả lời**:
1. Chia ảnh thành patches (16x16 pixels)
2. Flatten mỗi patch thành vector
3. Add positional embedding
4. Mỗi patch "attend" đến tất cả patches khác qua Q, K, V matrices
5. Tính attention weights: softmax(QK^T / √d)
6. Weighted sum của V theo attention weights
7. Cho phép model học global dependencies

#### 13. **Data augmentation bạn sử dụng là gì? Tại sao?**
**Trả lời** (trong `datasets_cls.py`):
- **RandomResizedCrop**: Mô phỏng các góc chụp khác nhau
- **RandomHorizontalFlip**: Lá có thể lật ngang tự nhiên
- **ColorJitter** (brightness, contrast, saturation): Điều kiện ánh sáng khác nhau
- **RandomRotation**: Góc quay của lá khi chụp
- Mục đích: Tăng diversity, giảm overfitting, model robust hơn

#### 14. **Early stopping hoạt động như thế nào trong dự án?**
**Trả lời** (trong `engine.py`):
```python
patience = 10  # Chờ 10 epochs
```
- Track validation F1-score mỗi epoch
- Nếu F1 không tăng sau 10 epochs liên tiếp → dừng training
- Lưu checkpoint của epoch có F1 cao nhất
- Tránh overfitting và tiết kiệm thời gian

#### 15. **Learning rate scheduler bạn dùng là gì? Giải thích.**
**Trả lời**: **CosineAnnealingLR**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```
- Learning rate giảm theo dạng cosine từ lr ban đầu về eta_min
- Smooth convergence, tránh oscillation
- Phù hợp cho vision tasks

#### 16. **Confidence threshold 60% có ý nghĩa gì?**
**Trả lời** (trong `validation.py`):
- **< 60%**: 🔴 Low confidence - Có thể KHÔNG phải lá lúa hoặc ảnh kém chất lượng
- **60-80%**: 🟡 Medium confidence - Chấp nhận được
- **≥ 80%**: 🟢 High confidence - Rất tin cậy

Lý do: Dựa trên phân tích validation set, model với confidence < 60% thường dự đoán sai hoặc ảnh không phải rice leaf.

#### 17. **Tại sao ViT cần pretrained weights nhiều hơn CNN?**
**Trả lời**:
- **CNN**: Có inductive bias mạnh (locality, translation invariance) → học tốt từ ít data
- **ViT**: Minimal inductive bias, coi ảnh như sequence → cần nhiều data để học pattern
- Pretrained trên ImageNet (1.2M ảnh) giúp ViT có starting point tốt
- Với small dataset (~1000 ảnh), pretrained ViT vượt trội CNN train from scratch

#### 18. **Giải thích Auto Color Normalization trong dự án.**
**Trả lời** (trong `color_normalization.py`):
```python
def auto_normalize_leaf(image):
    # Convert RGB → HSV
    # Adjust hue về xanh lá chuẩn (target_hue ~ 100-110°)
    # Adjust saturation về mức vừa phải
    # Adjust brightness về optimal range
```
- **Mục đích**: Chuẩn hóa màu sắc lá về điều kiện chuẩn
- **Lợi ích**: Giảm ảnh hưởng của ánh sáng, camera khác nhau → tăng accuracy
- **Trade-off**: Có thể mất thông tin màu sắc quan trọng của bệnh

#### 19. **Class imbalance được xử lý như thế nào?**
**Trả lời**:
- **Phân tích**: Dùng `dataset_stats.py` để visualize distribution
- **Stratified sampling**: Train/val/test split giữ tỷ lệ mỗi lớp
- **Data augmentation**: Tăng cường cho lớp thiểu số
- **Macro F1-score**: Đánh giá công bằng các lớp (không bị bias về lớp đa số)
- **Weighted sampling** (nếu cần): Sample nhiều hơn từ lớp thiểu số

#### 20. **Checkpoint lưu gì? Cấu trúc như thế nào?**
**Trả lời** (trong `engine.py`):
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "best_f1": best_f1,
    "config": {...}
}
```
- Lưu best checkpoint (highest F1) và last checkpoint
- Cho phép resume training hoặc inference

---

### 🔴 KHÓ (Câu 21-30) - Câu hỏi nâng cao & tình huống

#### 21. **Nếu model overfitting, bạn sẽ làm gì?**
**Trả lời**:
1. **Tăng regularization**:
   - Tăng weight decay (L2 regularization)
   - Thêm Dropout layers
2. **Data augmentation mạnh hơn**: Thêm MixUp, CutMix
3. **Early stopping**: Giảm patience
4. **Giảm model complexity**: Dùng model nhỏ hơn
5. **Thêm data**: Collect thêm ảnh thực tế
6. **Label smoothing**: Giảm overconfidence

#### 22. **Giải thích backpropagation trong Transformer.**
**Trả lời**:
1. Forward pass: Input → Multi-head attention → MLP → Output
2. Compute loss với ground truth
3. Backward pass:
   - Gradient flow qua softmax classification head
   - Qua LayerNorm và residual connections
   - Qua Multi-head attention (gradient của Q, K, V matrices)
   - Qua patch embedding và positional encoding
4. **Residual connections** giúp gradient flow tốt hơn (tránh vanishing gradient)
5. **LayerNorm** stabilize training

#### 23. **Tại sao dùng F1-score thay vì Accuracy làm metric chính?**
**Trả lời**:
- **Accuracy**: Có thể misleading khi imbalanced
  - VD: 90% healthy, 10% diseased → model dự đoán "healthy" cho tất cả vẫn có 90% accuracy
- **F1-score (macro)**:
  - Trung bình điều hòa của Precision và Recall
  - Đánh giá công bằng mọi lớp, kể cả thiểu số
  - F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Trong y tế/nông nghiệp**: Không bỏ sót bệnh (high recall) quan trọng

#### 24. **Giải thích gradient descent và momentum trong optimizer.**
**Trả lời**:
- **Gradient Descent**: θ = θ - lr × ∇L
- **Momentum** (trong Adam):
  ```
  m_t = β1 × m_{t-1} + (1-β1) × ∇L      # First moment (mean)
  v_t = β2 × v_{t-1} + (1-β2) × (∇L)²   # Second moment (variance)
  θ = θ - lr × m_t / (√v_t + ε)
  ```
- **β1=0.9, β2=0.999**: Smooth gradient, adaptive learning rate
- **AdamW**: Decouple weight decay khỏi gradient step

#### 25. **Transfer learning hoạt động như thế nào? Tại sao hiệu quả?**
**Trả lời**:
- **Cơ chế**:
  1. Load pretrained weights từ ImageNet
  2. Freeze early layers (học features cơ bản: edges, textures)
  3. Fine-tune later layers + classification head
- **Hiệu quả vì**:
  - Low-level features (edges, colors) tương tự giữa datasets
  - Không cần học lại từ đầu
  - Ít data vẫn converge tốt
- **Trong dự án**: CNN và ViT đều dùng pretrained → F1 ~87% với chỉ ~1000 ảnh

#### 26. **Batch size ảnh hưởng như thế nào đến training?**
**Trả lời**:
- **Batch size nhỏ (8-16)**:
  - ✅ Gradient noisy hơn → regularization effect, tránh sharp minima
  - ✅ Dùng ít VRAM
  - ❌ Slow, nhiều iterations
- **Batch size lớn (64-128)**:
  - ✅ Fast, stable gradient
  - ✅ Tận dụng GPU parallelism
  - ❌ Cần nhiều VRAM
  - ❌ Có thể converge đến sharp minima (generalize kém)
- **Trong dự án**: Batch size 16 (cân bằng speed và stability cho dataset nhỏ)

#### 27. **Nếu model bị bias về một lớp, làm sao fix?**
**Trả lời**:
1. **Root cause**: Imbalanced data hoặc lớp đó dễ phân biệt
2. **Solutions**:
   - **Weighted loss**: Gán trọng số cao cho lớp thiểu số
     ```python
     weights = 1 / class_counts
     criterion = nn.CrossEntropyLoss(weight=weights)
     ```
   - **Focal loss**: Focus vào hard examples
   - **Oversampling**: Duplicate minority class
   - **Class-balanced augmentation**: Aug mạnh hơn cho minority
   - **Two-stage training**: Train lại classification head với balanced batch

#### 28. **Giải thích Confusion Matrix và cách sử dụng.**
**Trả lời**:
```
              Predicted
           Blast  Healthy  ...
Actual  
Blast      80      5       ...   ← True Positives, False Negatives
Healthy     3     90       ...
```

**Phân tích**:
- **Diagonal**: Dự đoán đúng
- **Off-diagonal**: Confusion giữa các lớp
- **Pattern quan trọng**:
  - Nếu "healthy" thường bị nhầm thành "brown_spot" → 2 lớp này tương đồng cao
  - Nếu "blast" ít bị nhầm → model phân biệt tốt bệnh nghiêm trọng
  
**Actions**:
- Thu thập thêm data cho cặp bị confused
- Augmentation target vào cặp đó
- Feature engineering: tìm features phân biệt 2 lớp

#### 29. **Deployment: Làm sao optimize model cho production?**
**Trả lời**:
1. **Model Quantization**:
   - FP32 → FP16 or INT8
   - Giảm 2-4x size, tăng 2-3x speed
   - Trade-off: Accuracy giảm ~0.5-1%
   
2. **Model Pruning**:
   - Loại bỏ weights/neurons không quan trọng
   - Structured pruning: Loại channels/layers
   
3. **Knowledge Distillation**:
   - Train model nhỏ (student) học từ model lớn (teacher)
   - VD: ViT (teacher) → SmallCNN (student)
   
4. **ONNX Export**:
   ```python
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```
   - Cross-platform, tích hợp với TensorRT, OpenVINO
   
5. **Caching & Batching**:
   - Batch nhiều requests cùng lúc
   - Cache predictions cho ảnh đã thấy

#### 30. **Thiết kế pipeline end-to-end từ ảnh đến dự đoán.**
**Trả lời**:
```
1. IMAGE ACQUISITION
   ├─ User upload qua Gradio UI
   ├─ Validation: file format, size
   └─ Load to PIL Image

2. PREPROCESSING
   ├─ [Optional] Manual adjustments: brightness, contrast, rotation
   ├─ [Optional] Auto color normalization (HSV correction)
   ├─ Resize to 224x224
   ├─ ToTensor: [0,255] → [0,1]
   └─ Normalize: ImageNet mean/std

3. INFERENCE
   ├─ Load pretrained model (CNN or ViT)
   ├─ model.eval() + torch.no_grad()
   ├─ Forward pass: image → logits
   └─ Softmax: logits → probabilities

4. POST-PROCESSING
   ├─ ArgMax: get predicted class index
   ├─ Confidence check: threshold validation
   ├─ Top-K predictions: argsort probabilities
   └─ Disease status: healthy vs diseased

5. VISUALIZATION
   ├─ Display predicted class + confidence
   ├─ Bar chart: probability distribution
   ├─ Warning messages: low confidence
   └─ [Optional] Model comparison (CNN vs ViT)

6. OUTPUT
   ├─ Markdown formatted results
   ├─ Probability plots
   └─ [Optional] Save to JSON/CSV
```

**Critical considerations**:
- **Latency**: ~50-100ms cho ViT, ~15-20ms cho CNN
- **Error handling**: Invalid images, low confidence
- **Logging**: Track predictions for monitoring
- **Feedback loop**: Collect user feedback for retraining

---

## 🎯 Mẹo chuẩn bị phỏng vấn

### Hiểu sâu 3 khía cạnh:
1. **Lý thuyết ML**: Loss, optimizer, metrics, architectures
2. **Implementation**: Code structure, libraries, best practices  
3. **Domain knowledge**: Rice diseases, practical deployment

### Luôn có ví dụ cụ thể:
- Đừng chỉ nói "dùng data augmentation"
- Nói "dùng RandomHorizontalFlip với p=0.5 vì lá lúa có thể xuất hiện theo nhiều hướng"

### Chuẩn bị demo:
- Chạy Gradio app thành thạo
- Test với cả ảnh tốt và ảnh xấu
- So sánh CNN vs ViT trực tiếp

### Biết điểm mạnh/yếu:
- **Mạnh**: Dual model, visualization tốt, pipeline hoàn chỉnh
- **Yếu**: Dataset nhỏ, chưa deploy production, chưa mobile app

**Chúc bạn bảo vệ thành công! 🚀**

---

## 🧮 LÝ THUYẾT VỀ THUẬT TOÁN & ÁP DỤNG TRONG DỰ ÁN

### 📘 PHẦN 1: CNN (Convolutional Neural Network)

#### 1.1 Lý Thuyết Toán Học

**Convolution Operation**:
```
Output[i,j] = Σ Σ Input[i+m, j+n] × Kernel[m,n] + bias
             m n
```

**Ví dụ cụ thể**:
- Input: 224×224×3 (RGB image)
- Kernel: 3×3×3×64 (64 filters, mỗi filter 3×3 trên 3 channels)
- Output: 224×224×64 (với padding='same')

**Các thành phần chính**:

1. **Convolution Layer**:
   - Chiết xuất features cục bộ (edges, textures, patterns)
   - Shared weights → translation invariance
   - Formula: y = σ(W * x + b) 
     - W: filter weights
     - *: convolution operation
     - σ: activation (ReLU)

2. **Pooling Layer**:
   - Max pooling: f(x) = max(x_i) trong window
   - Down-sampling: giảm spatial dimensions
   - Tăng receptive field, giảm computation

3. **Activation Function (ReLU)**:
   - ReLU(x) = max(0, x)
   - Giải quyết vanishing gradient
   - Sparse activation (hiệu quả tính toán)

4. **Batch Normalization**:
   ```
   BN(x) = γ × (x - μ_batch) / √(σ²_batch + ε) + β
   ```
   - Normalize activations
   - Stabilize training, tăng learning rate được

#### 1.2 Áp Dụng Trong Dự Án - SmallCNN

**Architecture** (trong `src/models/cnn_small.py`):
```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes=5):
        # Block 1: 3×3 conv → BN → ReLU → MaxPool
        Conv2d(3, 32, kernel_size=3, padding=1)
        BatchNorm2d(32)
        ReLU()
        MaxPool2d(2, 2)  # 224→112
        
        # Block 2: 32→64 channels
        Conv2d(32, 64, 3, 1)
        # ... → 112→56
        
        # Block 3: 64→128 channels
        # ... → 56→28
        
        # Block 4: 128→256 channels
        # ... → 28→14
        
        # Global Average Pooling: 14×14×256 → 1×1×256
        AdaptiveAvgPool2d(1)
        
        # FC layer: 256 → 5 classes
        Linear(256, num_classes)
```

**Tại sao thiết kế như vậy?**:
- **4 blocks với increasing channels (32→64→128→256)**: 
  - Early layers: low-level features (edges, colors)
  - Later layers: high-level features (disease patterns)
  
- **Kernel size 3×3**: Standard, cân bằng receptive field và parameters

- **Global Average Pooling thay vì Flatten**:
  - Giảm overfitting (ít parameters hơn)
  - Translation invariance tốt hơn
  
- **Tổng parameters**: ~1.5MB (lightweight, fast inference ~15ms)

**Forward Pass Example**:
```
Input: [1, 3, 224, 224]      # Batch=1, RGB, 224×224
  ↓ Block1
[1, 32, 112, 112]            # 32 feature maps
  ↓ Block2  
[1, 64, 56, 56]
  ↓ Block3
[1, 128, 28, 28]
  ↓ Block4
[1, 256, 14, 14]
  ↓ Global Avg Pool
[1, 256, 1, 1] → [1, 256]
  ↓ FC
[1, 5]                        # Logits for 5 classes
```

---

### 📗 PHẦN 2: Vision Transformer (ViT)

#### 2.1 Lý Thuyết Toán Học

**Self-Attention Mechanism** (Core of Transformer):

1. **Input Transformation**:
   ```
   Q = X × W_Q    # Query
   K = X × W_K    # Key  
   V = X × W_V    # Value
   ```
   - X: input embeddings [N, D] (N patches, D dimensions)
   - W_Q, W_K, W_V: learned projection matrices

2. **Attention Scores**:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ```
   - QK^T: similarity scores giữa các patches [N, N]
   - /√d_k: scaling factor (d_k = dimension of K)
   - softmax: normalize thành xác suất
   - ×V: weighted sum theo importance

3. **Multi-Head Attention**:
   ```
   MultiHead(X) = Concat(head_1, ..., head_h) × W_O
   head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
   ```
   - h heads (thường h=8 hoặc 12)
   - Mỗi head học different aspects
   - Concat và project lại

4. **Transformer Block**:
   ```
   # Self-Attention
   X' = X + MultiHeadAttention(LayerNorm(X))
   
   # Feed-Forward Network
   X'' = X' + MLP(LayerNorm(X'))
   ```
   - Residual connections (X + ...)
   - LayerNorm trước attention/MLP
   - MLP: 2 FC layers với GELU activation

#### 2.2 Áp Dụng Trong Dự Án - ViT Small

**Architecture** (timm's `vit_small_patch16_224`):

```
Input Image: 224×224×3

1. PATCH EMBEDDING
   ├─ Chia ảnh: 14×14 patches (mỗi patch 16×16 pixels)
   ├─ Flatten: 14×14×(16×16×3) → 196×768
   └─ Linear projection: [196, 768]

2. POSITIONAL ENCODING
   ├─ Learnable position embeddings [196, 768]
   └─ X = Patch_Emb + Pos_Emb

3. [CLS] TOKEN
   ├─ Prepend special token for classification
   └─ X = [CLS; X] → [197, 768]

4. TRANSFORMER ENCODER (12 layers)
   For each layer:
   ├─ Multi-Head Attention (8 heads)
   │   ├─ Q, K, V projections
   │   ├─ Attention: softmax(QK^T/√96)V
   │   └─ Output projection
   ├─ Residual + LayerNorm
   ├─ MLP (768 → 3072 → 768)
   └─ Residual + LayerNorm

5. CLASSIFICATION HEAD
   ├─ Extract [CLS] token: [768]
   ├─ LayerNorm
   └─ Linear: [768] → [5 classes]
```

**Key Parameters**:
- Embedding dim: 768
- Num heads: 8 (each head: 768/8 = 96 dims)
- MLP ratio: 4× (768 → 3072 → 768)
- Depth: 12 transformer blocks
- Total params: ~22M → ~87MB checkpoint

**Tại sao ViT hiệu quả?**:

1. **Global receptive field ngay từ layer 1**:
   - CNN: receptive field tăng dần qua layers
   - ViT: mọi patch attend to all patches ngay từ đầu
   - → Học long-range dependencies tốt hơn

2. **Self-attention học adaptive**:
   - CNN: fixed kernel weights
   - ViT: attention weights change theo input
   - → Flexible hơn cho complex patterns

3. **Pretrained on large dataset**:
   - ImageNet-21k (14M images)
   - Learn powerful visual representations
   - Transfer tốt sang rice disease classification

**Trade-offs**:
- ✅ Accuracy cao hơn CNN (89.2% vs 87.3%)
- ✅ Better với complex diseases
- ❌ Slow hơn (50-100ms vs 15ms)
- ❌ Model size lớn (87MB vs 1.5MB)

---

### 📙 PHẦN 3: Optimization Algorithms

#### 3.1 Gradient Descent Variants

**1. Vanilla SGD** (Stochastic Gradient Descent):
```python
θ_t = θ_{t-1} - η × ∇L(θ_{t-1})
```
- η: learning rate
- ∇L: gradient của loss
- **Vấn đề**: Oscillation, slow convergence

**2. SGD with Momentum**:
```python
v_t = β × v_{t-1} + ∇L(θ)
θ_t = θ_{t-1} - η × v_t
```
- β: momentum coefficient (thường 0.9)
- v: velocity (exponential moving average of gradients)
- **Lợi ích**: Smooth trajectory, faster convergence

**3. Adam** (Adaptive Moment Estimation):
```python
# First moment (mean)
m_t = β1 × m_{t-1} + (1-β1) × ∇L

# Second moment (uncentered variance)
v_t = β2 × v_{t-1} + (1-β2) × (∇L)²

# Bias correction
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

# Update
θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)
```
- β1 = 0.9, β2 = 0.999, ε = 1e-8
- Adaptive learning rate cho mỗi parameter
- **Best for**: Deep neural networks

**4. AdamW** (Adam with Weight Decay):
```python
# Adam update
θ_t = θ_{t-1} - η × m̂_t / (√v̂_t + ε)

# THEN apply weight decay SEPARATELY
θ_t = θ_t - η × λ × θ_t
```
- λ: weight decay coefficient
- **Tại sao tách riêng?**: L2 regularization trong Adam bị coupled với adaptive learning rate
- **Better for**: Transformers, vision models

#### 3.2 Áp Dụng Trong Dự Án

**Training Configuration** (trong `configs/`):

```yaml
# CNN config (cls_cnn_small.yaml)
optimizer:
  type: AdamW
  lr: 1e-4              # Higher lr for CNN
  weight_decay: 1e-4    # L2 regularization
  betas: [0.9, 0.999]

# ViT config (cls_vit_s.yaml)  
optimizer:
  type: AdamW
  lr: 5e-5              # Lower lr for ViT (pretrained)
  weight_decay: 0.05    # Stronger regularization
  betas: [0.9, 0.999]
```

**Tại sao khác nhau?**:

1. **Learning Rate**:
   - CNN: 1e-4 (train from scratch → cần lr cao hơn)
   - ViT: 5e-5 (pretrained → fine-tune nhẹ nhàng)

2. **Weight Decay**:
   - CNN: 1e-4 (model nhỏ, risk overfitting thấp)
   - ViT: 0.05 (model lớn, cần regularize mạnh)

**Learning Rate Scheduler**:
```python
CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# LR decay theo cosine:
lr_t = eta_min + (lr_0 - eta_min) × (1 + cos(πt/T)) / 2
```

Example với lr_0=1e-4, T=50 epochs:
```
Epoch  0: lr = 1.00e-4
Epoch 12: lr = 7.07e-5  (giảm dần)
Epoch 25: lr = 5.00e-5  (mid-point)
Epoch 37: lr = 2.93e-5
Epoch 50: lr = 1.00e-6  (eta_min)
```

**Lợi ích**:
- Smooth convergence (không có sudden drops như StepLR)
- Exploration ở đầu (high lr) → exploitation ở cuối (low lr)

---

### 📕 PHẦN 4: Backpropagation Algorithm

#### 4.1 Lý Thuyết

**Chain Rule**:
```
∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
```

**Example với 1 FC layer**:
```
Forward:
z = W × x + b
y = σ(z)         # σ = activation (ReLU, softmax, etc.)
L = loss(y, y_true)

Backward:
∂L/∂W = ∂L/∂y × ∂y/∂z × ∂z/∂W
      = ∂L/∂y × σ'(z) × x^T

∂L/∂b = ∂L/∂y × σ'(z)

∂L/∂x = W^T × (∂L/∂y × σ'(z))  # Pass to previous layer
```

#### 4.2 Backprop Through CNN

**Convolution Layer**:
```
Forward: Y = Conv(X, W) + b

Backward:
∂L/∂W = Conv(X, ∂L/∂Y)           # Gradient w.r.t weights
∂L/∂X = ConvTranspose(∂L/∂Y, W)  # Gradient w.r.t input
```

**Max Pooling**:
```
Forward: Y = MaxPool(X)
Backward: ∂L/∂X[i] = {
    ∂L/∂Y[j]  if X[i] was the max
    0         otherwise
}
```

#### 4.3 Backprop Through Transformer

**Self-Attention**:
```
Forward:
S = QK^T / √d
A = softmax(S)
Y = A × V

Backward (simplified):
∂L/∂V = A^T × ∂L/∂Y
∂L/∂A = ∂L/∂Y × V^T
∂L/∂S = ∂softmax/∂S × ∂L/∂A
∂L/∂Q = ∂L/∂S × K / √d
∂L/∂K = ∂L/∂S^T × Q / √d
```

**Residual Connection**:
```
Forward: Y = X + F(X)
Backward: ∂L/∂X = ∂L/∂Y × (1 + ∂F/∂X)
```
- Gradient flow trực tiếp qua "1" → tránh vanishing

**LayerNorm**:
```
Forward: y = γ × (x - μ) / σ + β
Backward: cần tính ∂L/∂x qua normalization
```
- Phức tạp nhưng stable gradients

#### 4.4 Áp Dụng Trong Training Loop

**Trong `src/core/engine.py`**:
```python
def train_one_epoch(model, dataloader, criterion, optimizer):
    for images, labels in dataloader:
        # 1. FORWARD PASS
        logits = model(images)          # CNN hoặc ViT
        loss = criterion(logits, labels)  # CrossEntropyLoss
        
        # 2. BACKWARD PASS
        optimizer.zero_grad()           # Reset gradients
        loss.backward()                 # Backpropagation
        
        # 3. GRADIENT CLIPPING (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 4. OPTIMIZER STEP
        optimizer.step()                # Update weights
```

**CrossEntropyLoss Backward**:
```
Loss = -Σ y_true[i] × log(softmax(logits)[i])

∂Loss/∂logits = softmax(logits) - y_true
```
- Đơn giản và numerically stable

---

### 📒 PHẦN 5: Regularization Techniques

#### 5.1 Weight Decay (L2 Regularization)

**Lý thuyết**:
```
Loss_total = Loss_data + λ/2 × Σ w²

∂Loss_total/∂w = ∂Loss_data/∂w + λ × w
```

**Trong AdamW** (dự án dùng):
```python
# Standard update
θ = θ - lr × gradient

# With weight decay
θ = θ × (1 - lr × λ) - lr × gradient
```

**Trong dự án**:
- CNN: λ = 1e-4 (mild regularization)
- ViT: λ = 0.05 (strong regularization vì model lớn)

#### 5.2 Dropout

**Lý thuyết**:
```
Training: y = DropOut(x, p=0.5) = {
    0        with probability p
    x/(1-p)  with probability 1-p
}

Inference: y = x  (no dropout)
```

**Không dùng trong dự án này** vì:
- CNN: BatchNorm đã regularize tốt
- ViT: Pretrained + weight decay đủ

#### 5.3 Data Augmentation (Implicit Regularization)

**Trong `datasets_cls.py`**:
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

**Hiệu ứng**: Mỗi epoch model thấy "ảnh mới" → generalize tốt hơn

---

### 🎓 PHẦN 6: Ứng Dụng Tổng Hợp

#### Quy Trình Training End-to-End

```python
# INITIALIZATION
model = SmallCNN(num_classes=5)  # hoặc ViT
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# TRAINING LOOP
for epoch in range(num_epochs):
    # TRAINING PHASE
    model.train()
    for batch in train_loader:
        images, labels = batch
        
        # Forward: sử dụng CNN/ViT algorithms
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward: backpropagation algorithm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # VALIDATION PHASE
    model.eval()
    with torch.no_grad():  # Không tính gradient
        for batch in val_loader:
            logits = model(images)
            # Compute metrics...
    
    # Learning rate decay
    scheduler.step()
    
    # Early stopping check
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break  # Stop training
```

**Kết nối các thuật toán**:
1. **CNN/ViT**: Feature extraction
2. **Softmax**: Convert logits → probabilities
3. **CrossEntropy**: Loss computation
4. **Backpropagation**: Compute gradients
5. **AdamW**: Update weights
6. **CosineAnnealing**: Adjust learning rate

---

## 🔬 Câu Hỏi Bổ Sung Về Thuật Toán

#### 31. **Giải thích chi tiết softmax và tại sao dùng trong classification.**
**Trả lời**:
```python
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

**Tính chất**:
- Output: xác suất [0, 1], tổng = 1
- Differentiable: backprop được
- Amplify differences: class có logit cao → probability cao hơn nhiều

**Trong dự án**:
```python
logits = model(image)  # [1, 5]: [2.3, 5.1, 1.2, 0.8, 3.4]
probs = softmax(logits) # [0.04, 0.66, 0.01, 0.01, 0.12]
predicted = argmax(probs)  # Class 1 (index 1)
```

#### 32. **Residual connection giải quyết vấn đề gì?**
**Trả lời**:
```python
# Without residual: y = F(x)
# Gradient: ∂L/∂x = ∂L/∂y × ∂F/∂x

# With residual: y = x + F(x)
# Gradient: ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)
```

**Vanishing Gradient Problem**:
- Deep networks: gradient × × × → 0
- Residual: luôn có "1" trong gradient → flow tốt

**Trong ViT**: Mỗi transformer block có 2 residual connections

#### 33. **Batch Normalization vs Layer Normalization?**
**Trả lời**:

**Batch Norm** (CNN dùng):
```
μ_batch = mean(X across batch dimension)
BN(x) = (x - μ_batch) / σ_batch
```
- Normalize theo batch → require large batch size
- Dùng trong CNN (spatial dimensions tương đồng)

**Layer Norm** (ViT dùng):
```
μ_layer = mean(X across feature dimension)  
LN(x) = (x - μ_layer) / σ_layer
```
- Normalize theo features → independent of batch size
- Better cho Transformers (sequence length vary)

**Chúc bạn hiểu sâu về thuật toán và áp dụng tốt! 🚀**

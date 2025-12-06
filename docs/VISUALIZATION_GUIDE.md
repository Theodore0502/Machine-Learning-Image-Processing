# Hướng dẫn sử dụng Visualization So sánh CNN vs ViT

## 📊 Tổng quan

Công cụ visualization mới cho phép so sánh trực quan giữa CNN và ViT model với nhiều góc nhìn khác nhau, tất cả trong một hình ảnh duy nhất.

## 🎨 Cách sử dụng

### 1. Hiển thị visualization (không lưu file)

```bash
python -m src.visualization.model_comparison --image temp/test1.jpg
```

Visualization sẽ hiển thị trực tiếp trên màn hình.

### 2. Lưu visualization vào file

```bash
python -m src.visualization.model_comparison \
  --image temp/test1.jpg \
  --save outputs/cnn_vs_vit_comparison.png
```

File sẽ được lưu với định dạng PNG, resolution 150 DPI (phù hợp in ấn).

### 3. Sử dụng custom checkpoints

```bash
python -m src.visualization.model_comparison \
  --image temp/test1.jpg \
  --cnn_checkpoint path/to/cnn_model.pt \
  --vit_checkpoint path/to/vit_model.pt \
  --save comparison.png
```

## 📈 Nội dung Visualization

Visualization bao gồm **6 subplots** chính:

### Hàng 1 (Top row):
1. **Ảnh gốc** (trái): Hiển thị ảnh input được dự đoán
2. **Dự đoán CNN** (giữa): 
   - Tên lớp dự đoán (phông chữ lớn, màu đỏ)
   - Độ tin cậy (%)
   - Thời gian inference (ms)
3. **Dự đoán ViT** (phải):
   - Tên lớp dự đoán (phông chữ lớn, màu xanh)
   - Độ tin cậy (%)
   - Thời gian inference (ms)

### Hàng 2 (Middle row):
4. **Top-5 Predictions Bar Chart** (chiếm toàn bộ chiều ngang):
   - So sánh xác suất dự đoán top-5 của cả 2 models
   - CNN: màu đỏ, ViT: màu xanh
   - Có nhãn % trên mỗi cột

### Hàng 3 (Bottom row):
5. **All Classes Comparison** (trái):
   - Horizontal bar chart so sánh tất cả các lớp
   - Dễ nhìn thấy sự khác biệt giữa 2 models

6. **Inference Speed** (giữa):
   - Bar chart so sánh tốc độ
   - Hiển thị speedup factor (CNN nhanh hơn bao nhiêu lần)

7. **Agreement Analysis** (phải):
   - ✓ ĐỒNG THUẬN hoặc ⚠ BẤT ĐỒNG
   - Chi tiết dự đoán của từng model
   - Chênh lệch độ tin cậy
   - **Khuyến nghị** thông minh dựa trên kết quả

## 🎯 Ứng dụng

### Cho Báo cáo cuối kỳ:
- **Trực quan hóa chất lượng cao**: PNG 150 DPI phù hợp in ấn
- **So sánh toàn diện**: Thể hiện hiểu biết về cả 2 architectures
- **Professional**: Layout đẹp, dễ đọc, màu sắc rõ ràng

### Cho Demo:
- **Nhanh chóng**: 1 lệnh tạo tất cả visualizations
- **Tương tác**: Có thể chạy real-time không cần lưu file
- **Giải thích được**: Có khuyến nghị và phân tích

### Cho Phân tích:
- **Tìm disagreement cases**: Nhìn thấy khi nào 2 models không đồng ý
- **Hiểu trade-offs**: Speed vs Accuracy rõ ràng
- **Debug**: Xem distribution của probabilities

## 💡 Tips

1. **Batch processing**: Tạo visualization cho nhiều ảnh bằng shell script:
```bash
for img in temp/*.jpg; do
    python -m src.visualization.model_comparison \
        --image "$img" \
        --save "outputs/comparison_$(basename $img)"
done
```

2. **Integration với báo cáo**: Embed trực tiếp vào LaTeX/Word
```latex
\includegraphics[width=\textwidth]{outputs/cnn_vs_vit_comparison.png}
```

3. **Presentation**: Use as slide background cho phần so sánh models

## 📦 Dependencies

Script sử dụng:
- `matplotlib`: Vẽ biểu đồ
- `seaborn`: Style đẹp hơn
- `numpy`: Xử lý arrays
- `torch`, `timm`: Load models
- `PIL`: Xử lý ảnh

Tất cả đều đã có trong `requirements.txt`.

## ⚙️ Customization

Để tùy chỉnh màu sắc hoặc layout, edit file `src/visualization/model_comparison.py`:

```python
# Đổi màu
DEFAULT_CONFIG = {
    "cnn": {
        "color": "#FF6B6B",  # Màu đỏ cho CNN (thay đổi tại đây)
        ...
    },
    "vit": {
        "color": "#4ECDC4",  # Màu xanh cho ViT (thay đổi tại đây)
        ...
    }
}

# Đổi size figure
plt.rcParams['figure.figsize'] = (16, 10)  # Width, Height
plt.rcParams['figure.dpi'] = 100  # Resolution
```

## 🎉 Kết quả mẫu

Sau khi chạy lệnh, bạn sẽ thấy output:
```
🔧 Device: cuda
📚 Số lớp: 5

📦 Đang load models...
  - CNN: runs/cls_cnn_small/weights/cnn_small_best.pt
  - ViT: runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt
✅ Đã load xong models

🔮 Đang dự đoán...
✅ Hoàn thành dự đoán

🎨 Đang tạo visualization...
✅ Đã lưu visualization: outputs/cnn_vs_vit_comparison.png
```

File output: `outputs/cnn_vs_vit_comparison.png` (~400-500 KB)

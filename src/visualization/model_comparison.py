# src/visualization/model_comparison.py
"""
Visualization tool for comparing CNN vs ViT model predictions and performance.

Usage:
    python -m src.visualization.model_comparison --image temp/test1.jpg
    python -m src.visualization.model_comparison --image temp/test1.jpg --save outputs/comparison.png
"""
import argparse
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from src.models.cnn_small import SmallCNN


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 100


# Default configurations
DEFAULT_CONFIG = {
    "cnn": {
        "checkpoint": "runs/cls_cnn_small/weights/cnn_small_best.pt",
        "model_name": "cnn_small",
        "img_size": 224,
        "color": "#FF6B6B",  # Red
        "label": "CNN (SmallCNN)"
    },
    "vit": {
        "checkpoint": "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt",
        "model_name": "vit_small_patch16_224",
        "img_size": 224,
        "color": "#4ECDC4",  # Teal
        "label": "ViT (Small)"
    }
}

# Vietnamese class name mapping
CLASS_NAME_VI = {
    "bacterial_blight": "Đạo ôn lúa",
    "blast": "Cháy lá",
    "brown_spot": "Đốm nâu",
    "healthy": "Khỏe mạnh",
    "tungro": "Vàng lùn"
}


def load_labels(path):
    """Load class labels from file."""
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def build_transform(img_size):
    """Build image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_model(checkpoint, model_name, num_classes, device):
    """Load trained model from checkpoint."""
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    try:
        sd = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(checkpoint, map_location=device)
    
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    
    model.load_state_dict(sd)
    return model.to(device).eval()


def predict_with_model(image_path, model, transform, device):
    """Make prediction with a model and return detailed results."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return probs, inference_time, img


def create_comparison_visualization(image_path, cnn_results, vit_results, class_names, save_path=None):
    """Create comprehensive comparison visualization."""
    cnn_probs, cnn_time, original_img = cnn_results
    vit_probs, vit_time, _ = vit_results
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Get Vietnamese class names
    class_names_vi = [CLASS_NAME_VI.get(c, c) for c in class_names]
    
    # Color scheme
    cnn_color = DEFAULT_CONFIG["cnn"]["color"]
    vit_color = DEFAULT_CONFIG["vit"]["color"]
    
    # =========================
    # 1. Original Image (top left)
    # =========================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img)
    ax1.set_title(f"Ảnh gốc\n{os.path.basename(image_path)}", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # =========================
    # 2. Prediction Comparison (top middle & right)
    # =========================
    cnn_pred_idx = np.argmax(cnn_probs)
    vit_pred_idx = np.argmax(vit_probs)
    
    # CNN prediction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.6, class_names_vi[cnn_pred_idx], 
             ha='center', va='center', fontsize=20, fontweight='bold', color=cnn_color)
    ax2.text(0.5, 0.4, f"Độ tin cậy: {cnn_probs[cnn_pred_idx]:.2%}", 
             ha='center', va='center', fontsize=14)
    ax2.text(0.5, 0.2, f"Thời gian: {cnn_time:.2f}ms", 
             ha='center', va='center', fontsize=12, style='italic')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Dự đoán CNN", fontsize=14, fontweight='bold', color=cnn_color)
    ax2.axis('off')
    
    # ViT prediction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, class_names_vi[vit_pred_idx], 
             ha='center', va='center', fontsize=20, fontweight='bold', color=vit_color)
    ax3.text(0.5, 0.4, f"Độ tin cậy: {vit_probs[vit_pred_idx]:.2%}", 
             ha='center', va='center', fontsize=14)
    ax3.text(0.5, 0.2, f"Thời gian: {vit_time:.2f}ms", 
             ha='center', va='center', fontsize=12, style='italic')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title("Dự đoán ViT", fontsize=14, fontweight='bold', color=vit_color)
    ax3.axis('off')
    
    # =========================
    # 3. Top-5 Predictions Bar Chart (middle row, spanning all columns)
    # =========================
    ax4 = fig.add_subplot(gs[1, :])
    
    top5_indices = np.argsort(-cnn_probs)[:5]
    x_pos = np.arange(len(top5_indices))
    width = 0.35
    
    cnn_top5 = cnn_probs[top5_indices]
    vit_top5 = vit_probs[top5_indices]
    labels_top5 = [class_names_vi[i] for i in top5_indices]
    
    bars1 = ax4.bar(x_pos - width/2, cnn_top5 * 100, width, 
                    label='CNN', color=cnn_color, alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, vit_top5 * 100, width, 
                    label='ViT', color=vit_color, alpha=0.8)
    
    ax4.set_xlabel('Lớp bệnh', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Độ tin cậy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('So sánh Top-5 Dự đoán', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels_top5, rotation=15, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 2:  # Only show if > 2%
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)
    
    # =========================
    # 4. All Classes Probability Comparison (bottom left)
    # =========================
    ax5 = fig.add_subplot(gs[2, 0])
    
    y_pos = np.arange(len(class_names))
    ax5.barh(y_pos - 0.2, cnn_probs * 100, 0.4, 
             label='CNN', color=cnn_color, alpha=0.8)
    ax5.barh(y_pos + 0.2, vit_probs * 100, 0.4, 
             label='ViT', color=vit_color, alpha=0.8)
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(class_names_vi, fontsize=10)
    ax5.set_xlabel('Xác suất (%)', fontsize=11, fontweight='bold')
    ax5.set_title('So sánh tất cả các lớp', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(axis='x', alpha=0.3)
    ax5.set_xlim(0, 100)
    
    # =========================
    # 5. Inference Speed Comparison (bottom middle)
    # =========================
    ax6 = fig.add_subplot(gs[2, 1])
    
    times = [cnn_time, vit_time]
    colors = [cnn_color, vit_color]
    labels = ['CNN', 'ViT']
    
    bars = ax6.bar(labels, times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Thời gian (ms)', fontsize=11, fontweight='bold')
    ax6.set_title('So sánh tốc độ Inference', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{time_val:.2f}ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add speedup annotation
    speedup = vit_time / cnn_time
    ax6.text(0.5, max(times) * 0.7, 
             f'CNN nhanh hơn {speedup:.1f}x',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # =========================
    # 6. Agreement/Disagreement Analysis (bottom right)
    # =========================
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    agree = (cnn_pred_idx == vit_pred_idx)
    conf_diff = abs(cnn_probs[cnn_pred_idx] - vit_probs[vit_pred_idx])
    
    # Status box
    if agree:
        status_color = 'lightgreen'
        status_text = '✓ ĐỒNG THUẬN'
        detail_text = f'Cả 2 model đều dự đoán:\n"{class_names_vi[cnn_pred_idx]}"'
    else:
        status_color = 'lightyellow'
        status_text = '⚠ BẤT ĐỒNG'
        detail_text = f'CNN: {class_names_vi[cnn_pred_idx]}\nViT: {class_names_vi[vit_pred_idx]}'
    
    # Draw status box
    ax7.add_patch(mpatches.FancyBboxPatch(
        (0.1, 0.6), 0.8, 0.3,
        boxstyle="round,pad=0.05",
        facecolor=status_color,
        edgecolor='black',
        linewidth=2
    ))
    ax7.text(0.5, 0.75, status_text, 
             ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Detail text
    ax7.text(0.5, 0.45, detail_text, 
             ha='center', va='top', fontsize=12)
    
    # Confidence difference
    ax7.text(0.5, 0.25, f'Chênh lệch độ tin cậy: {conf_diff:.2%}', 
             ha='center', va='center', fontsize=11, style='italic')
    
    # Recommendation
    if agree and cnn_probs[cnn_pred_idx] > 0.8:
        recommendation = '→ Nên dùng CNN (nhanh & tin cậy)'
    elif not agree and conf_diff > 0.2:
        winner = "ViT" if vit_probs[vit_pred_idx] > cnn_probs[cnn_pred_idx] else "CNN"
        recommendation = f'→ Nên tin {winner} (độ tin cậy cao hơn)'
    else:
        recommendation = '→ Cân nhắc thêm hoặc dùng ensemble'
    
    ax7.text(0.5, 0.08, recommendation, 
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.set_title('Phân tích', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('So sánh Dự đoán: CNN vs Vision Transformer (ViT)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu visualization: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="🎨 CNN vs ViT Comparison Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--image", type=str, required=True,
                       help="Đường dẫn đến ảnh cần dự đoán")
    parser.add_argument("--save", type=str,
                       help="Đường dẫn lưu hình visualization (nếu không có sẽ hiển thị)")
    parser.add_argument("--cnn_checkpoint", type=str,
                       help="Custom CNN checkpoint path")
    parser.add_argument("--vit_checkpoint", type=str,
                       help="Custom ViT checkpoint path")
    parser.add_argument("--labels_file", type=str, default="data/splits/labels.txt",
                       help="File chứa tên các lớp")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    # Load class names
    class_names = load_labels(args.labels_file)
    num_classes = len(class_names)
    print(f"📚 Số lớp: {num_classes}")
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Không tìm thấy ảnh: {args.image}")
        return
    
    # Load models
    print("\n📦 Đang load models...")
    
    # CNN
    cnn_cfg = DEFAULT_CONFIG["cnn"]
    cnn_ckpt = args.cnn_checkpoint or cnn_cfg["checkpoint"]
    if not os.path.exists(cnn_ckpt):
        print(f"❌ Không tìm thấy CNN checkpoint: {cnn_ckpt}")
        return
    print(f"  - CNN: {cnn_ckpt}")
    cnn_model = load_model(cnn_ckpt, cnn_cfg["model_name"], num_classes, device)
    cnn_transform = build_transform(cnn_cfg["img_size"])
    
    # ViT
    vit_cfg = DEFAULT_CONFIG["vit"]
    vit_ckpt = args.vit_checkpoint or vit_cfg["checkpoint"]
    if not os.path.exists(vit_ckpt):
        print(f"❌ Không tìm thấy ViT checkpoint: {vit_ckpt}")
        return
    print(f"  - ViT: {vit_ckpt}")
    vit_model = load_model(vit_ckpt, vit_cfg["model_name"], num_classes, device)
    vit_transform = build_transform(vit_cfg["img_size"])
    
    print("✅ Đã load xong models\n")
    
    # Make predictions
    print("🔮 Đang dự đoán...")
    cnn_results = predict_with_model(args.image, cnn_model, cnn_transform, device)
    vit_results = predict_with_model(args.image, vit_model, vit_transform, device)
    print("✅ Hoàn thành dự đoán\n")
    
    # Create visualization
    print("🎨 Đang tạo visualization...")
    create_comparison_visualization(
        args.image, 
        cnn_results, 
        vit_results, 
        class_names,
        save_path=args.save
    )
    
    if not args.save:
        print("\n💡 Tip: Thêm --save <path> để lưu hình thay vì hiển thị")


if __name__ == "__main__":
    main()

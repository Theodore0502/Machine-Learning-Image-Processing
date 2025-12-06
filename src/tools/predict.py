# src/tools/predict.py
"""
Unified prediction interface supporting both CNN and ViT models.
Default: CNN (for course requirement)
Usage:
    python -m src.tools.predict --image temp/test1.jpg
    python -m src.tools.predict --image temp/test1.jpg --model_type vit
    python -m src.tools.predict --image temp/test1.jpg --model_type both
    python -m src.tools.predict --image_dir temp/ --model_type cnn
"""
import argparse
import os
import time
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from src.models.cnn_small import SmallCNN


# Cấu hình mặc định
DEFAULT_CONFIG = {
    "cnn": {
        "checkpoint": "runs/cls_cnn_small/weights/cnn_small_best.pt",
        "model_name": "cnn_small",
        "img_size": 224,
    },
    "vit": {
        "checkpoint": "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt",
        "model_name": "vit_small_patch16_224",
        "img_size": 224,
    }
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


def predict_single(image_path, model, transform, class_names, device, model_type):
    """Predict single image and return results."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    inference_time = (time.time() - start_time) * 1000  # ms
    
    pred_idx = int(np.argmax(probs))
    pred_name = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    return {
        "model_type": model_type,
        "predicted_class": pred_name,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "inference_time_ms": inference_time,
        "top_predictions": [(class_names[i], float(probs[i])) 
                           for i in np.argsort(-probs)[:min(5, len(class_names))]]
    }


def print_prediction(result, image_path):
    """Print prediction results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"📷 Ảnh: {image_path}")
    print(f"🤖 Model: {result['model_type'].upper()}")
    print(f"{'='*60}")
    print(f"✅ Dự đoán: {result['predicted_class']}")
    print(f"📊 Độ tin cậy: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print(f"⏱️  Thời gian: {result['inference_time_ms']:.2f}ms")
    print(f"\nTop 5 dự đoán:")
    for i, (cls, prob) in enumerate(result['top_predictions'], 1):
        bar = "█" * int(prob * 30)
        print(f"  {i}. {cls:<20} {prob:.4f} {bar}")
    
    # Kiểm tra nếu có lớp "healthy"
    top_class_names = [c.lower() for c, _ in result['top_predictions']]
    if any("healthy" in c for c in top_class_names):
        is_diseased = "healthy" not in result['predicted_class'].lower()
        status = "CÓ BỆNH ⚠️" if is_diseased else "KHỎE MẠNH ✓"
        print(f"\n🌾 Tình trạng: {status}")


def compare_predictions(cnn_result, vit_result, image_path):
    """Compare predictions from both models."""
    print(f"\n{'='*70}")
    print(f"📊 SO SÁNH DỰ ĐOÁN: CNN vs ViT")
    print(f"📷 Ảnh: {image_path}")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<10} {'Dự đoán':<20} {'Độ tin cậy':<15} {'Thời gian (ms)':<15}")
    print("-" * 70)
    print(f"{'CNN':<10} {cnn_result['predicted_class']:<20} "
          f"{cnn_result['confidence']:.4f} ({cnn_result['confidence']*100:.1f}%)    "
          f"{cnn_result['inference_time_ms']:<.2f}")
    print(f"{'ViT':<10} {vit_result['predicted_class']:<20} "
          f"{vit_result['confidence']:.4f} ({vit_result['confidence']*100:.1f}%)    "
          f"{vit_result['inference_time_ms']:<.2f}")
    
    # Phân tích
    agree = cnn_result['predicted_class'] == vit_result['predicted_class']
    print(f"\n{'✅ ĐỒNG THUẬN' if agree else '⚠️ BẤT ĐỒNG'}: ", end="")
    if agree:
        print(f"Cả hai model đều dự đoán '{cnn_result['predicted_class']}'")
    else:
        print(f"CNN dự đoán '{cnn_result['predicted_class']}', "
              f"ViT dự đoán '{vit_result['predicted_class']}'")
    
    # So sánh tốc độ
    speedup = vit_result['inference_time_ms'] / cnn_result['inference_time_ms']
    print(f"⚡ Tốc độ: CNN nhanh hơn ViT {speedup:.1f}x")
    
    # Khuyến nghị
    print(f"\n💡 Khuyến nghị:")
    if agree and cnn_result['confidence'] > 0.8:
        print(f"   → Dùng CNN (nhanh hơn, cả 2 model đồng thuận)")
    elif not agree:
        conf_diff = abs(cnn_result['confidence'] - vit_result['confidence'])
        if conf_diff > 0.2:
            winner = "ViT" if vit_result['confidence'] > cnn_result['confidence'] else "CNN"
            print(f"   → Nên tin {winner} (độ tin cậy cao hơn rõ rệt)")
        else:
            print(f"   → Nên xem xét thêm (2 model không đồng thuận và độ tin cậy gần bằng nhau)")
    else:
        print(f"   → Dùng ViT nếu cần độ chính xác cao, CNN nếu cần tốc độ")


def main():
    parser = argparse.ArgumentParser(
        description="🌾 Rice Leaf Disease Prediction - Dual Model Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dự đoán với CNN (mặc định - yêu cầu môn học)
  python -m src.tools.predict --image temp/test1.jpg
  
  # Dự đoán với ViT (độ chính xác cao hơn)
  python -m src.tools.predict --image temp/test1.jpg --model_type vit
  
  # So sánh cả 2 model
  python -m src.tools.predict --image temp/test1.jpg --model_type both
  
  # Dự đoán batch nhiều ảnh
  python -m src.tools.predict --image_dir temp/ --model_type cnn
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Đường dẫn đến ảnh đơn lẻ")
    input_group.add_argument("--image_dir", type=str, help="Thư mục chứa nhiều ảnh")
    
    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["cnn", "vit", "both"],
        default="cnn",
        help="Loại model: cnn (nhanh, mặc định), vit (chính xác), both (so sánh)"
    )
    
    # Optional overrides
    parser.add_argument("--cnn_checkpoint", type=str, help="Custom CNN checkpoint path")
    parser.add_argument("--vit_checkpoint", type=str, help="Custom ViT checkpoint path")
    parser.add_argument("--labels_file", type=str, default="data/splits/labels.txt",
                       help="File chứa tên các lớp")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    class_names = load_labels(args.labels_file)
    num_classes = len(class_names)
    print(f"📚 Số lớp: {num_classes} - {class_names}")
    
    # Load models
    models = {}
    if args.model_type in ["cnn", "both"]:
        cnn_cfg = DEFAULT_CONFIG["cnn"]
        cnn_ckpt = args.cnn_checkpoint or cnn_cfg["checkpoint"]
        if not os.path.exists(cnn_ckpt):
            print(f"❌ Không tìm thấy checkpoint CNN: {cnn_ckpt}")
            print(f"   Hãy train model trước: python src/train.py --task cls --config configs/cls_cnn_small.yaml")
            return
        print(f"📦 Đang load CNN model từ {cnn_ckpt}...")
        models["cnn"] = {
            "model": load_model(cnn_ckpt, cnn_cfg["model_name"], num_classes, device),
            "transform": build_transform(cnn_cfg["img_size"]),
            "config": cnn_cfg
        }
    
    if args.model_type in ["vit", "both"]:
        vit_cfg = DEFAULT_CONFIG["vit"]
        vit_ckpt = args.vit_checkpoint or vit_cfg["checkpoint"]
        if not os.path.exists(vit_ckpt):
            print(f"❌ Không tìm thấy checkpoint ViT: {vit_ckpt}")
            print(f"   Hãy train model trước: python src/train.py --task cls --config configs/cls_vit_s.yaml")
            return
        print(f"📦 Đang load ViT model từ {vit_ckpt}...")
        models["vit"] = {
            "model": load_model(vit_ckpt, vit_cfg["model_name"], num_classes, device),
            "transform": build_transform(vit_cfg["img_size"]),
            "config": vit_cfg
        }
    
    print(f"✅ Đã load {len(models)} model(s)\n")
    
    # Get image paths
    if args.image:
        image_paths = [args.image]
    else:
        img_dir = Path(args.image_dir)
        image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + \
                     list(img_dir.glob("*.jpeg"))
        image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print("❌ Không tìm thấy ảnh nào!")
        return
    
    print(f"📸 Số ảnh cần dự đoán: {len(image_paths)}\n")
    
    # Process images
    all_results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"⚠️ Không tìm thấy: {img_path}")
            continue
        
        results = {}
        for model_type, model_data in models.items():
            result = predict_single(
                img_path,
                model_data["model"],
                model_data["transform"],
                class_names,
                device,
                model_type
            )
            results[model_type] = result
        
        # Display results
        if args.model_type == "both":
            compare_predictions(results["cnn"], results["vit"], img_path)
        else:
            print_prediction(results[args.model_type], img_path)
        
        all_results.append({"image": img_path, "predictions": results})
    
    # Save to JSON if requested
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Đã lưu kết quả vào: {args.output}")


if __name__ == "__main__":
    main()

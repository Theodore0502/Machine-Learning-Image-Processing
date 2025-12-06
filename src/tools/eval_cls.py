# src/tools/eval_cls.py  (clean print, không dùng classification_report string)
import argparse, os, torch, csv, re
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from src.models.cnn_small import SmallCNN


def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def build_val_tf(sz):
    return transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )


def _split_line(line: str):
    s = line.strip()
    if not s or s.startswith("#"):
        return "", None
    # tách theo ký tự phân cách CUỐI CÙNG (space/tab/comma) trước nhãn số ở cuối
    m = re.search(r"[ \t,](\d+)$", s)
    if not m:
        raise ValueError(f"Bad line: {line}")
    lab = int(m.group(1))
    p = s[: m.start()].strip()
    return p, lab


def _normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    if os.path.isabs(p):
        return p
    low = p.lower()
    if low.startswith("public/rice_cls/"):
        p = "data/" + p[len("public/") :]  # -> data/rice_cls/...
    elif low.startswith("rice_cls/"):
        p = "data/" + p  # -> data/rice_cls/...
    return p


def parse_split(fp):
    paths, labels = [], []
    with open(fp, "r", encoding="utf-8") as f:
        for raw in f:
            p, lab = _split_line(raw)
            if p == "" and lab is None:
                continue
            p = _normalize_path(p)
            paths.append(p)
            labels.append(int(lab))
    return paths, labels


def load_model(ckpt, model_name, num_classes, device):
    if model_name == "cnn_small":
        model = SmallCNN(num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    # cố gắng load an toàn nếu PyTorch hỗ trợ
    try:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)

    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    # Simplified interface with model_type
    ap.add_argument("--model_type", type=str, choices=["cnn", "vit"],
                   help="Model type: cnn or vit (auto-detects checkpoint)")
    # Original explicit parameters (for backward compatibility)
    ap.add_argument("--ckpt")
    ap.add_argument("--model_name")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--split_file", default="data/splits/test_cls.txt")
    ap.add_argument("--labels_file", default="data/splits/labels.txt")
    ap.add_argument("--out_csv", default="eval_preds.csv")
    args = ap.parse_args()

    # Auto-detect checkpoint and model_name from model_type
    if args.model_type:
        configs = {
            "cnn": {
                "ckpt": "runs/cls_cnn_small/weights/cnn_small_best.pt",
                "model_name": "cnn_small"
            },
            "vit": {
                "ckpt": "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt",
                "model_name": "vit_small_patch16_224"
            }
        }
        cfg = configs[args.model_type]
        args.ckpt = args.ckpt or cfg["ckpt"]
        args.model_name = args.model_name or cfg["model_name"]
    
    # Validate required parameters
    if not args.ckpt or not args.model_name:
        ap.error("Either provide --model_type OR both --ckpt and --model_name")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = load_class_map(args.labels_file)
    num_classes = len(class_names)
    tfm = build_val_tf(args.img_size)
    model = load_model(args.ckpt, args.model_name, num_classes, device)

    paths, y_true = parse_split(args.split_file)
    y_pred = []

    print(f"🔧 Device: {device}")
    print(f"🔢 Số lớp: {num_classes} - {class_names}")
    print(f"📂 Số mẫu cần đánh giá: {len(paths)}\n")

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # === TÍNH METRICS ===
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # === IN GỌN BẢNG METRIC ===
    print("📊 KẾT QUẢ ĐÁNH GIÁ\n")
    header = f"{'Lớp':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Sup':>7}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        print(
            f"{name:<18} "
            f"{prec[i]:7.4f} "
            f"{rec[i]:7.4f} "
            f"{f1[i]:7.4f} "
            f"{sup[i]:7d}"
        )

    print("\nTổng quan:")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro avg F1  : {f1_macro:.4f}")
    print(f"  Weighted F1   : {f1_weighted:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    print("\n🔢 Confusion matrix (hàng = true, cột = pred):")
    print(cm)

    # Lưu CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred"])
        for p, t, pr in zip(paths, y_true, y_pred):
            w.writerow([p, class_names[int(t)], class_names[int(pr)]])
    print(f"\n✅ Saved: {args.out_csv}")


if __name__ == "__main__":
    main()

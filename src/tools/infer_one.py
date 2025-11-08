# src/tools/infer_one.py
import argparse, os, torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from src.models.cnn_small import SmallCNN

def load_labels(p):
    with open(p, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def build_tf(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    if os.path.isabs(p): return p
    low = p.lower()
    if low.startswith("public/rice_cls/"): p = "data/" + p[len("public/"):]
    elif low.startswith("rice_cls/"):       p = "data/" + p
    return p

def resolve_path(p: str, root: str=".") -> str:
    p = normalize_path(p)
    if os.path.isabs(p): return p
    if os.path.exists(p): return p
    return os.path.join(root, p).replace("\\","/")

def load_model(ckpt, model_name, num_classes, device):
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    try:
        sd = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    return model.to(device).eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--labels_file", default="data/splits/labels.txt")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = load_labels(args.labels_file)
    num_classes = len(class_names)
    model = load_model(args.ckpt, args.model_name, num_classes, device)
    tfm = build_tf(args.img_size)

    img_path = resolve_path(args.img, ".")
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        topk = min(args.topk, num_classes)
        topk_idx = np.argsort(-probs)[:topk]

    print(f"Image: {img_path}")
    print(f"Pred:  {pred_name} (idx={pred_idx})  prob={probs[pred_idx]:.4f}")
    print("Top-k:")
    for i in topk_idx:
        print(f"  {class_names[i]:<20} {probs[i]:.4f}")

    # Nếu labels có 'healthy', in thêm nhận định có bệnh/không
    if "healthy" in [n.lower() for n in class_names]:
        healthy_idx = [n.lower() for n in class_names].index("healthy")
        is_diseased = pred_idx != healthy_idx
        print(f"Diseased?: {'YES' if is_diseased else 'NO'}")

if __name__ == "__main__":
    main()

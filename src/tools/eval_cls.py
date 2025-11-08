# src/tools/eval_cls.py  (robust split: space/comma/tab + path normalize)
import argparse, os, torch, csv, re
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.models.cnn_small import SmallCNN

def load_class_map(path):
    with open(path,"r",encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def build_val_tf(sz):
    return transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

def _split_line(line: str):
    s = line.strip()
    if not s or s.startswith("#"):
        return "", None
    # tách theo "kí tự phân cách CUỐI CÙNG" trong [space, tab, comma]
    # để không phá đường dẫn có khoảng trắng ở giữa
    m = re.search(r"[ \t,](\d+)$", s)
    if not m:
        raise ValueError(f"Bad line: {line}")
    lab = int(m.group(1))
    p = s[:m.start()].strip()
    return p, lab

def _normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\","/")
    if os.path.isabs(p):
        return p
    # Chuẩn hoá các prefix hay gặp
    low = p.lower()
    if low.startswith("public/rice_cls/"):
        p = "data/" + p[len("public/"):]      # -> data/rice_cls/...
    elif low.startswith("rice_cls/"):
        p = "data/" + p                       # -> data/rice_cls/...
    return p

def parse_split(fp):
    paths, labels = [], []
    with open(fp,"r",encoding="utf-8") as f:
        for raw in f:
            p, lab = _split_line(raw)
            if p == "" and lab is None:
                continue
            p = _normalize_path(p)
            paths.append(p); labels.append(int(lab))
    return paths, labels

def load_model(ckpt, model_name, num_classes, device):
    if model_name=="cnn_small":
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
    model.load_state_dict(sd); model.to(device).eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--labels_file", required=True)
    ap.add_argument("--out_csv", default="eval_preds.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = load_class_map(args.labels_file); num_classes = len(class_names)
    tfm = build_val_tf(args.img_size)
    model = load_model(args.ckpt, args.model_name, num_classes, device)

    paths, y_true = parse_split(args.split_file)
    y_pred = []

    for p in paths:
        # nếu path là tương đối, để nguyên (data_root là '.')
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()
        y_pred.append(pred)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path","true","pred"])
        for p,t,pr in zip(paths,y_true,y_pred):
            w.writerow([p,class_names[t],class_names[pr]])
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()

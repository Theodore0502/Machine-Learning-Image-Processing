# src/tools/preview_transforms.py
import argparse, os
from PIL import Image
import torch
from torchvision import transforms, utils

def build_train_tf(img_size):
    # Lưu ý: ToTensor trước ColorJitter & RandomErasing (đều chạy được trên Tensor)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # --> chuyển sang Tensor sớm
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def build_val_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def denorm(x):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (x*std+mean).clamp(0,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--n", type=int, default=6, help="số ảnh augment để ghép grid")
    ap.add_argument("--out", default="preview_aug.png")
    args = ap.parse_args()

    img = Image.open(args.img).convert("RGB")
    train_tf = build_train_tf(args.img_size)
    val_tf   = build_val_tf(args.img_size)

    base = val_tf(img)
    # tạo n biến thể augment khác nhau
    aug_list = [train_tf(img) for _ in range(args.n)]
    samples = [base] + aug_list
    samples = torch.stack([denorm(s) for s in samples], 0)  # đưa về [0,1] để save

    # ghép grid
    nrow = (args.n // 2) + 1
    grid = utils.make_grid(samples, nrow=nrow, padding=4)
    utils.save_image(grid, args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()

# Upgraded training script (AMP/TF32/accumulation/mixup/early-stop)
import argparse, yaml, os
import random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from src.data.datasets_cls import RiceClsDataset
from src.models.cnn_small import SmallCNN
@torch.no_grad()
def accuracy_top1(outputs, targets):
    _, pred = outputs.max(1)
    correct = pred.eq(targets).sum().item()
    return 100.0 * correct / targets.size(0)
@torch.no_grad()
def f1_macro(outputs, targets, num_classes):
    _, pred = outputs.max(1)
    f1_per_class = []
    for c in range(num_classes):
        tp = ((pred == c) & (targets == c)).sum().item()
        fp = ((pred == c) & (targets != c)).sum().item()
        fn = ((pred != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_per_class.append(f1)
    return float(sum(f1_per_class) / len(f1_per_class))
def build_transforms(img_size, aug_cfg):
    color = bool(aug_cfg.get("color", True))
    rand_erasing = float(aug_cfg.get("random_erasing", 0.0))
    tfs = [transforms.Resize((img_size, img_size)),
           transforms.RandomHorizontalFlip()]
    if color:
        tfs.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
    tfs.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    if rand_erasing > 0:
        tfs.append(transforms.RandomErasing(p=rand_erasing, value='random'))
    return transforms.Compose(tfs)
def build_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
def train_one_epoch(model, loader, optimizer, criterion, device, num_classes,
                    scaler, use_amp, accumulation_steps, mixup_fn=None):
    model.train()
    loss_sum = acc_sum = f1_sum = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            imgs, labels = mixup_fn(imgs, labels)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if mixup_fn is None:
            acc_sum += accuracy_top1(outputs, labels)
            f1_sum  += f1_macro(outputs, labels, num_classes)
        else:
        # labels từ Mixup là soft (one-hot pha trộn). Lấy argmax của NHÃN (không phải của output)
            if labels.dtype == torch.float:
                hard_targets = labels.argmax(dim=1)
            else:
                hard_targets = labels
                acc_sum += accuracy_top1(outputs, hard_targets)
                f1_sum  += f1_macro(outputs, hard_targets, num_classes)
        loss_sum += loss.item() * accumulation_steps
        n_batches += 1
    return loss_sum/n_batches, acc_sum/n_batches, f1_sum/n_batches
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum = acc_sum = f1_sum = 0.0
    n_batches = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        acc_sum += accuracy_top1(outputs, labels)
        f1_sum  += f1_macro(outputs, labels, num_classes)
        n_batches += 1
    return loss_sum/n_batches, acc_sum/n_batches, f1_sum/n_batches
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def train_cls(cfg):
    print("[train][cls] Using config:", cfg)
    data_root   = cfg.get("data_root", "data")
    train_split = cfg.get("train_split", "data/splits/train_cls.txt")
    val_split   = cfg.get("val_split",   "data/splits/val_cls.txt")
    model_name  = cfg.get("model", "vit_small_patch16_224")
    img_size    = int(cfg.get("img_size", 224))
    epochs      = int(cfg.get("epochs", 20))
    batch_size  = int(cfg.get("batch_size", 8))
    lr          = float(cfg.get("lr", 3e-4))
    weight_decay= float(cfg.get("weight_decay", 0.05))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    num_classes = int(cfg.get("num_classes", 4))
    accumulation_steps = int(cfg.get("accumulation_steps", 4))
    seed = int(cfg.get("seed", 1337))
    use_amp = bool(cfg.get("amp", True))
    use_tf32 = bool(cfg.get("tf32", True))
    cudnn_bench = bool(cfg.get("cudnn_benchmark", True))
    dl_cfg = cfg.get("dataloader", {})
    num_workers = int(dl_cfg.get("num_workers", 8))
    pin_memory  = bool(dl_cfg.get("pin_memory", True))
    prefetch_factor = int(dl_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(dl_cfg.get("persistent_workers", True)) and num_workers > 0
    aug_cfg = cfg.get("augment", {})
    mixup_alpha = float(aug_cfg.get("mixup", 0.2))
    cutmix_alpha = float(aug_cfg.get("cutmix", 0.2))
    use_mix = (mixup_alpha > 0.0) or (cutmix_alpha > 0.0)
    save_dir = cfg.get("save_dir", f"runs/{model_name}_{img_size}")
    os.makedirs(save_dir, exist_ok=True)
    weights_dir = os.path.join(save_dir, "weights"); os.makedirs(weights_dir, exist_ok=True)
    best_path = os.path.join(weights_dir, f"{model_name}_best.pt")
    patience = int(cfg.get("early_stop", {}).get("patience", 5))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = cudnn_bench
    set_seed(seed)
    t_train = build_transforms(img_size, aug_cfg)
    t_val   = build_val_transforms(img_size)
    dtrain = RiceClsDataset(data_root, train_split, transform=t_train)
    dval   = RiceClsDataset(data_root, val_split,   transform=t_val)
    train_loader = DataLoader(dtrain, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor if num_workers>0 else None,
                              persistent_workers=persistent_workers)
    val_loader   = DataLoader(dval, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              prefetch_factor=prefetch_factor if num_workers>0 else None,
                              persistent_workers=persistent_workers)
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    if use_mix:
        criterion_train = SoftTargetCrossEntropy()
        criterion_val = nn.CrossEntropyLoss()  # Validation luôn dùng hard labels
        mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
                         prob=1.0, switch_prob=0.5, mode='batch',
                         label_smoothing=0.0, num_classes=num_classes)
    else:
        criterion_train = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        criterion_val = criterion_train
        mixup_fn = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_f1 = -1.0
    es_counter = 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion_train, device, num_classes,
            scaler, use_amp, accumulation_steps, mixup_fn
        )
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion_val, device, num_classes)
        scheduler.step()
        print(f"[ep {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.2f}% f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.2f}% f1 {va_f1:.3f} | lr {scheduler.get_last_lr()[0]:.6f}")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({'model': model.state_dict(),
                        'num_classes': num_classes,
                        'img_size': img_size,
                        'model_name': model_name}, best_path)
            print(f"  -> saved best to {best_path} (f1={best_f1:.3f})")
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= patience:
                print(f"Early stopping at epoch {ep} (no val_f1 improve for {patience} epochs)." )
                break
def train_seg(_cfg):
    print("[train][seg] Placeholder. Ưu tiên hoàn tất classification trước.")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cls","seg"], required=True)
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.task == "cls":
        train_cls(cfg)
    else:
        train_seg(cfg)
if __name__ == "__main__":
    main()

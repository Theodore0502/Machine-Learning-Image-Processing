# src/tools/gradcam.py  — supports CNN (4D) and ViT (3D tokens)
import argparse, os, re, torch, cv2, numpy as np
from torchvision import transforms
from PIL import Image
import timm
from src.models.cnn_small import SmallCNN

# ---------- path helpers ----------
def _normalize_path(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    if os.path.isabs(p):
        return p
    low = p.lower()
    if low.startswith("public/rice_cls/"):
        p = "data/" + p[len("public/"):]
    elif low.startswith("rice_cls/"):
        p = "data/" + p
    return p

def _resolve_path(p: str, data_root: str) -> str:
    p = _normalize_path(p)
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(data_root, p).replace("\\", "/")

def _path_from_split(split_file: str, index: int, data_root: str) -> str:
    with open(split_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not lines:
        raise RuntimeError("Empty split file.")
    idx = max(0, min(index, len(lines)-1))
    s = lines[idx]
    m = re.search(r"[ \t,](\d+)$", s)
    p = s[:m.start()].strip() if m else s
    return _resolve_path(p, data_root)

# ---------- model / grad-cam ----------
def load_model(ckpt_path, model_name, num_classes, device):
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd); model.to(device).eval()
    return model

def pick_layer(model, name):
    if name == "cnn_small": return model.features[-2]      # last BN before GAP
    if hasattr(model, "layer4"): return model.layer4[-1]   # ResNet-like
    if hasattr(model, "blocks"):  return model.blocks[-1]  # ViT block (tokens B,N,C)
    raise RuntimeError("Chỉ định layer Grad-CAM chưa hỗ trợ.")

class GradCAM:
    """
    Works for:
      - CNN: activation/gradient are 4D (B,C,H,W)
      - ViT: activation/gradient are 3D (B,N,C). We drop CLS token and reshape to (B,H,W).
    """
    def __init__(self, model, target_layer, model_name, grid_hw=None):
        self.model = model
        self.model_name = model_name
        self.grid_hw = grid_hw  # (H, W) for ViT patch grid
        self.grad = None
        self.act  = None
        target_layer.register_forward_hook(self._fh)
        target_layer.register_full_backward_hook(self._bh)

    def _fh(self, m, i, o): self.act = o.detach()
    def _bh(self, m, gi, go): self.grad = go[0].detach()

    def _cam_cnn(self):
        # act, grad: (B,C,H,W)
        w = self.grad.mean(dim=(2,3), keepdim=True)              # (B,C,1,1)
        cam = (w * self.act).sum(dim=1, keepdim=True)            # (B,1,H,W)
        cam = torch.relu(cam).squeeze(0).squeeze(0).cpu().numpy()  # (H,W)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

    def _cam_vit(self):
        # act, grad: (B,N,C). N = 1 + H*W (CLS + patch tokens)
        B, N, C = self.act.shape
        w = self.grad.mean(dim=2, keepdim=True)      # (B,N,1)
        cam_tokens = (w * self.act).sum(dim=2)       # (B,N)
        # drop CLS token
        cam_tokens = cam_tokens[:, 1:]               # (B, H*W)
        H, W = self.grid_hw
        cam = cam_tokens.reshape(B, H, W)            # (B,H,W)
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

    def __call__(self):
        if self.act is None or self.grad is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")
        if self.act.dim() == 4 and self.grad.dim() == 4:
            return self._cam_cnn()
        if self.act.dim() == 3 and self.grad.dim() == 3 and self.grid_hw is not None:
            return self._cam_vit()
        raise RuntimeError(f"Unsupported shapes: act={tuple(self.act.shape)}, grad={tuple(self.grad.shape)}")

# ---------- image I/O ----------
def preprocess(p, sz):
    tfm = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    img = Image.open(p).convert("RGB")
    return tfm(img).unsqueeze(0), np.array(img)

def overlay(img_uint8, cam):
    h,w,_ = img_uint8.shape
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (w,h))
    return (0.4*heat + 0.6*img_uint8).astype(np.uint8)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--img", default="")
    ap.add_argument("--split_file", default="")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out", default="gradcam_vis.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model_name, args.num_classes, device)
    layer = pick_layer(model, args.model_name)

    # compute ViT grid if available
    grid_hw = None
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        gs = model.patch_embed.grid_size  # (H, W)
        if isinstance(gs, (tuple, list)) and len(gs) == 2:
            grid_hw = (int(gs[0]), int(gs[1]))

    cam = GradCAM(model, layer, args.model_name, grid_hw=grid_hw)

    # pick image
    if args.img:
        img_path = _resolve_path(args.img, args.data_root)
    elif args.split_file:
        img_path = _path_from_split(args.split_file, args.index, args.data_root)
    else:
        raise RuntimeError("Provide --img or --split_file.")

    # forward + backward
    x, raw = preprocess(img_path, args.img_size); x = x.to(device)
    x.requires_grad_(True)
    logits = model(x); pred = logits.argmax(1).item()
    loss = logits[0, pred]; model.zero_grad(set_to_none=True); loss.backward()

    # build CAM & save
    cam_map = cam()
    vis = overlay(raw, cam_map)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.out}. Pred={pred} | Img={img_path} | Grid={grid_hw}")

if __name__ == "__main__":
    main()

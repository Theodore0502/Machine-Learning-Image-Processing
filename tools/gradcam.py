# src/tools/gradcam.py
import torch, cv2, numpy as np
from torchvision import transforms
import timm
from src.models.cnn_small import SmallCNN

def load_model(ckpt_path, model_name, num_classes, device):
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd["model"])
    model.to(device).eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grad = None
        self.act  = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.act = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, logits, target_class):
        # weights: GAP over gradients
        weights = self.grad.mean(dim=(2,3), keepdim=True)  # (B,C,1,1)
        cam = (weights * self.act).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

def preprocess(img_path, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0), np.array(img)

def overlay_cam(img_uint8, cam):
    h, w, _ = img_uint8.shape
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (w, h))
    overlay = (0.4*heat + 0.6*img_uint8).astype(np.uint8)
    return overlay

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--img", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model_name, args.num_classes, device)

    # Chọn layer cuối cùng của backbone (tùy model)
    # Với cnn_small:
    target_layer = model.features[-2] if hasattr(model, "features") else None
    assert target_layer is not None, "Không tìm thấy layer mục tiêu cho Grad-CAM."

    cam_engine = GradCAM(model, target_layer)
    x, raw = preprocess(args.img, args.img_size)
    x = x.to(device)

    x.requires_grad_(True)
    logits = model(x)
    pred = logits.argmax(dim=1).item()

    loss = logits[0, pred]
    model.zero_grad(set_to_none=True)
    loss.backward()

    cam = cam_engine(logits, pred)
    vis = overlay_cam(raw, cam)

    out_path = "gradcam_vis.png"
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM to {out_path}. Pred class = {pred}")

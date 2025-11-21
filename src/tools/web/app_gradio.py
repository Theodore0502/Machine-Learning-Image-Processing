import os
from typing import List

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import cv2

from src.tools.image_editor.operations import apply_all
from src.tools.gradcam import GradCAM, pick_layer

# =========================
# CONFIG
# =========================
MODEL_NAME = "vit_small_patch16_224"
CKPT_PATH = "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt"
LABELS_FILE = "data/splits/labels.txt"
IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD LABELS
# =========================
def load_labels(labels_path: str) -> List[str]:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Check again data/splits/labels.txt."
        )
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


CLASS_NAMES = load_labels(LABELS_FILE)
NUM_CLASSES = len(CLASS_NAMES)


# =========================
# LOAD MODEL + GRADCAM
# =========================
def load_model(model_name: str, ckpt_path: str, num_classes: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model file not found: {ckpt_path}\n"
            f"Make sure you have finished training and the file vit_small_patch16_224_best.pt exists."
        )

    print(f"🔄 Loading model: {model_name} from {ckpt_path}")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "model" in raw:
        print("📌 Checkpoint is custom -> using raw['model']")
        state_dict = raw["model"]
    else:
        print("📌 Checkpoint is pure state_dict")
        state_dict = raw

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("⚠ Missing keys:", missing)
    print("⚠ Unexpected keys:", unexpected)

    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
    return model


model = load_model(MODEL_NAME, CKPT_PATH, NUM_CLASSES)

# Grad-CAM extractor
target_layer = pick_layer(model, MODEL_NAME)

grid_hw = None
if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
    gs = model.patch_embed.grid_size
    if isinstance(gs, (tuple, list)):
        grid_hw = (int(gs[0]), int(gs[1]))
    else:
        grid_hw = (int(gs[0]), int(gs[1]))

cam_extractor = GradCAM(model, target_layer, MODEL_NAME, grid_hw=grid_hw)


# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# =========================
# CUSTOM OVERLAY: xanh = tập trung cao, đỏ = thấp
# =========================
def overlay_green_focus(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    img_uint8: ảnh RGB gốc (H,W,3) uint8
    cam: map Grad-CAM (h_cam, w_cam), giá trị bất kỳ (sẽ được chuẩn hoá)
    alpha: độ đậm heatmap (0..1)
    """
    h, w, _ = img_uint8.shape

    cam_resized = cv2.resize(cam, (w, h))
    cam_min, cam_max = cam_resized.min(), cam_resized.max()

    if cam_max - cam_min < 1e-8:
        cam_norm = np.zeros_like(cam_resized)
    else:
        cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)

    cam_norm = np.clip(cam_norm, 0.0, 1.0)

    # lấy ngưỡng top 20% => vùng này sẽ tô xanh
    q = 0.8
    th = np.quantile(cam_norm, q)
    mask_high = cam_norm >= th

    heat = np.zeros_like(img_uint8, dtype=np.float32)
    # mặc định: đỏ (low focus)
    heat[..., 0] = 255.0  # R
    heat[..., 1] = 0.0    # G
    heat[..., 2] = 0.0    # B
    # vùng tập trung cao: xanh lá
    heat[mask_high, 0] = 0.0
    heat[mask_high, 1] = 255.0

    img_f = img_uint8.astype(np.float32)
    out = alpha * heat + (1.0 - alpha) * img_f
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# =========================
# EDIT ONLY (PREVIEW)
# =========================
def edit_image(
    img,
    brightness,
    contrast,
    hue_shift,
    sat_scale,
    val_scale,
    angle,
    flip_h,
    flip_v,
):
    if img is None:
        return None

    edited = apply_all(
        img,
        brightness=brightness,
        contrast=contrast,
        hue_shift=hue_shift,
        sat_scale=sat_scale,
        val_scale=val_scale,
        angle=angle,
        flip_h=flip_h,
        flip_v=flip_v,
    )
    return edited


# =========================
# PREDICT + GRADCAM
# =========================
def predict_from_controls(
    img,
    brightness,
    contrast,
    hue_shift,
    sat_scale,
    val_scale,
    angle,
    flip_h,
    flip_v,
):
    if img is None:
        return "❗ Please upload a rice leaf image!", None

    edited = apply_all(
        img,
        brightness=brightness,
        contrast=contrast,
        hue_shift=hue_shift,
        sat_scale=sat_scale,
        val_scale=val_scale,
        angle=angle,
        flip_h=flip_h,
        flip_v=flip_v,
    )

    x = transform(edited).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    # top-1
    top_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[top_idx]
    conf = probs[top_idx] * 100

    # top-k
    k = min(4, len(CLASS_NAMES))
    top_indices = probs.argsort()[::-1][:k]
    lines = [
        f"- **{CLASS_NAMES[i]}**: {probs[i]*100:.2f}%"
        for i in top_indices
    ]
    topk_text = "\n".join(lines)

    result_md = f"""
### 🌾 Prediction Result

**Predicted disease:** `{pred_label}`  
**Confidence:** `{conf:.2f}%`

**Top-{k} classes:**
{topk_text}
"""

    # Grad-CAM
    loss = logits[0, top_idx]
    model.zero_grad(set_to_none=True)
    loss.backward()

    cam_map = cam_extractor()  # (H_cam, W_cam)

    raw = np.array(edited.convert("RGB"))
    vis = overlay_green_focus(raw, cam_map, alpha=0.6)
    cam_pil = Image.fromarray(vis)

    return result_md, cam_pil


# =========================
# GRADIO UI
# =========================
def build_app():
    with gr.Blocks(title="Rice Leaf Health + Image Editing") as demo:
        gr.Markdown("## 🌾 Rice Leaf Health – Image Editing + Disease Prediction (with Grad-CAM)")

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Rice Leaf Image (Upload)", type="pil")

                brightness = gr.Slider(0, 100, value=50, step=1,
                                       label="Brightness (0 = darkest, 100 = brightest)")
                contrast   = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                       label="Contrast")
                hue_shift  = gr.Slider(-180, 180, value=0, step=10,
                                       label="Hue")
                sat_scale  = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                       label="Saturation")
                val_scale  = gr.Slider(0.0, 3.0, value=1.0, step=0.1,
                                       label="Value")
                angle      = gr.Dropdown(
                    choices=[0, 90, 180, 270],
                    value=0,
                    label="Rotation"
                )
                flip_h     = gr.Checkbox(value=False, label="Flip Horizontal")
                flip_v     = gr.Checkbox(value=False, label="Flip Vertical")

                btn = gr.Button("🔍 Predict Disease")

            with gr.Column():
                orig_img   = gr.Image(label="Original Image")
                preview_img = gr.Image(label="Edited Preview")
                heatmap_img = gr.Image(label="Model Attention (Green = high, Red = low)")
                output_text = gr.Markdown(label="Prediction")

        # nhóm tất cả input controls để feed vào các hàm
        controls = [
            img_input,
            brightness,
            contrast,
            hue_shift,
            sat_scale,
            val_scale,
            angle,
            flip_h,
            flip_v,
        ]

        # hiển thị lại ảnh gốc mỗi lần upload
        img_input.change(
            fn=lambda x: x,
            inputs=img_input,
            outputs=orig_img,
        )

        # bất cứ control nào đổi -> update preview ảnh chỉnh
        for c in controls:
            c.change(
                fn=edit_image,
                inputs=controls,
                outputs=preview_img,
            )

        # bấm nút -> dự đoán + Grad-CAM overlay
        btn.click(
            fn=predict_from_controls,
            inputs=controls,
            outputs=[output_text, heatmap_img],
        )

        return demo


app = build_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, debug=True)

import os
from typing import List

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

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
            f"Không tìm thấy file labels: {labels_path}\n"
            f"Kiểm tra lại data/splits/labels.txt."
        )
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


CLASS_NAMES = load_labels(LABELS_FILE)
NUM_CLASSES = len(CLASS_NAMES)


# =========================
# LOAD MODEL
# =========================
def load_model(model_name: str, ckpt_path: str, num_classes: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Không tìm thấy file model: {ckpt_path}\n"
            f"Bạn hãy chắc là đã train xong và file vit_small_patch16_224_best.pt tồn tại."
        )

    print(f"🔄 Loading model: {model_name} from {ckpt_path}")

    # tạo model ViT đúng cấu hình
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    # load checkpoint (custom: có key 'model')
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "model" in raw:
        print("📌 Checkpoint dạng custom -> dùng raw['model']")
        state_dict = raw["model"]
    else:
        print("📌 Checkpoint là state_dict thuần")
        state_dict = raw

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("⚠ Missing keys:", missing)
    print("⚠ Unexpected keys:", unexpected)

    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded thành công!")
    return model


model = load_model(MODEL_NAME, CKPT_PATH, NUM_CLASSES)


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
# PREDICT FUNCTION
# =========================
def predict(img: Image.Image):
    if img is None:
        return "❗ Hãy tải lên một ảnh lá lúa!", None

    # chuyển ảnh sang tensor
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # top-1
    top_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[top_idx]
    conf = float(probs[top_idx] * 100)

    # top-k (ở đây k = min(4, num_classes))
    k = min(4, len(CLASS_NAMES))
    top_indices = probs.argsort()[::-1][:k]

    lines = [
        f"- **{CLASS_NAMES[i]}**: {probs[i]*100:.2f}%"
        for i in top_indices
    ]
    topk_text = "\n".join(lines)

    result_md = f"""
### 🌾 Kết quả dự đoán

**Bệnh dự đoán:** `{pred_label}`  
**Độ tin cậy:** `{conf:.2f}%`

**Top-{k} class:**
{topk_text}
"""

    # trả về text + ảnh gốc (để hiển thị bên cạnh)
    return result_md, img


# =========================
# GRADIO UI
# =========================
def build_app():
    with gr.Blocks(title="Rice Leaf Disease Classification") as demo:
        gr.Markdown(
            """
# 🌿 Rice Leaf Disease Classification (ViT)

Tải ảnh lá lúa lên để mô hình dự đoán loại bệnh.
- Model: `vit_small_patch16_224`
- Ảnh được resize về 224x224, chuẩn hóa theo ImageNet.
            """
        )

        with gr.Row():
            with gr.Column():
                img_input = gr.Image(
                    label="Tải ảnh lá lúa",
                    type="pil"
                )
                btn = gr.Button("🔍 Dự đoán bệnh")
            with gr.Column():
                output_text = gr.Markdown(label="Kết quả dự đoán")
                output_img = gr.Image(label="Ảnh đã tải lên")

        btn.click(
            fn=predict,
            inputs=img_input,
            outputs=[output_text, output_img],
        )

        return demo


app = build_app()

if __name__ == "__main__":
    # chạy trên mọi interface, port 7860
    app.launch(server_name="0.0.0.0", server_port=7860, debug=True)

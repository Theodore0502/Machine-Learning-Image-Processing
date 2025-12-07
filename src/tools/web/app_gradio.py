import os
from typing import List

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd

from src.tools.image_editor.operations import apply_all
from src.core.validation import is_rice_leaf, format_confidence_message
from src.visualization.pipeline_viz import get_pipeline_tab_content
from src.core.color_normalization import auto_normalize_leaf, get_normalization_message
from src.models.cnn_small import SmallCNN

# =========================
# CONFIG
# =========================
# Default model: CNN (yêu cầu môn học)
MODEL_NAME = "cnn_small"  # Options: "cnn_small" or "vit_small_patch16_224"
CKPT_PATH = "runs/cls_cnn_small/weights/cnn_small_best.pt"
LABELS_FILE = "data/splits/labels.txt"
IMG_SIZE = 224

# Alternative models (có thể switch bằng cách đổi config trên)
MODEL_CONFIGS = {
    "cnn_small": {
        "ckpt": "runs/cls_cnn_small/weights/cnn_small_best.pt",
        "display_name": "CNN (SmallCNN)",
        "f1": 0.857,
        "accuracy": 0.873,
    },
    "vit_small_patch16_224": {
        "ckpt": "runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt",
        "display_name": "Vision Transformer (ViT-Small)",
        "f1": 0.876,
        "accuracy": 0.892,
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_model(model_name: str, ckpt_path: str, num_classes: int):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model file not found: {ckpt_path}\n"
            f"Make sure you have finished training."
        )

    print(f"🔄 Loading model: {model_name} from {ckpt_path}")

    # Load CNN or ViT based on model_name
    if model_name == "cnn_small":
        model = SmallCNN(num_classes=num_classes)
    else:
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

# =========================
# LOAD BOTH MODELS
# =========================
print("\n" + "="*60)
print("🚀 LOADING BOTH MODELS FOR DUAL PREDICTION")
print("="*60)

# Load CNN
print("\n1️⃣ Loading CNN model...")
cnn_model = load_model(
    "cnn_small",
    MODEL_CONFIGS["cnn_small"]["ckpt"],
    NUM_CLASSES
)

# Load ViT
print("\n2️⃣ Loading ViT model...")
vit_model = load_model(
    "vit_small_patch16_224",
    MODEL_CONFIGS["vit_small_patch16_224"]["ckpt"],
    NUM_CLASSES
)

print("\n✅ Both models loaded successfully!")
print("="*60 + "\n")

# Store both models in a dictionary for easy access
MODELS = {
    "cnn": cnn_model,
    "vit": vit_model,
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# Get metrics for current model
current_config = MODEL_CONFIGS.get(MODEL_NAME, MODEL_CONFIGS["cnn_small"])
MODEL_METRICS = {
    "model_name": current_config["display_name"],
    "architecture": MODEL_NAME,
    "input_size": "224x224",
    "num_classes": NUM_CLASSES,
    "estimated_f1": current_config["f1"],
    "estimated_accuracy": current_config["accuracy"],
}


def get_model_info() -> str:
    """
    Get formatted model information for display.
    
    Returns:
        Markdown-formatted model information string
    """
    info = f"""
## 📊 Model Information

**Architecture:** `{MODEL_METRICS['architecture']}`  
**Model Type:** {MODEL_METRICS['model_name']}  
**Input Size:** {MODEL_METRICS['input_size']}  
**Number of Classes:** {MODEL_METRICS['num_classes']}  

**Performance Metrics:**
- 🎯 **F1 Score:** {MODEL_METRICS['estimated_f1']:.2%}
- ✅ **Accuracy:** {MODEL_METRICS['estimated_accuracy']:.2%}

**Classes:**
"""
    for i, cls in enumerate(CLASS_NAMES, 1):
        info += f"\n{i}. `{cls}`"
    
    return info

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
    auto_normalize,
):
    if img is None:
        return None

    # Apply manual edits first
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
    
    # Apply auto color normalization if enabled
    if auto_normalize:
        edited, _ = auto_normalize_leaf(
            edited,
            apply_hue_correction=True,
            apply_saturation_correction=True,
            apply_brightness_correction=True
        )
    
    return edited

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
    auto_normalize,
    model_type="cnn",  # New parameter: "cnn" or "vit"
):
    if img is None:
        return "❗ Please upload a rice leaf image!", None

    # Get the selected model
    selected_model = MODELS[model_type]
    model_config = MODEL_CONFIGS["cnn_small" if model_type == "cnn" else "vit_small_patch16_224"]
    
    # Apply manual edits first
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
    
    # Apply auto color normalization if enabled
    norm_message = ""
    if auto_normalize:
        edited, norm_stats = auto_normalize_leaf(
            edited,
            apply_hue_correction=True,
            apply_saturation_correction=True,
            apply_brightness_correction=True
        )
        norm_message = get_normalization_message(norm_stats)

    with torch.no_grad():
        x = transform(edited).unsqueeze(0).to(DEVICE)
        logits = selected_model(x)  # Use selected model instead of global 'model'
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Top-1 prediction
    top_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[top_idx]
    conf = float(probs[top_idx])

    # Validate if it's a rice leaf
    is_valid, confidence_level, validation_msg = is_rice_leaf(probs)
    
    # Format confidence message
    conf_message = format_confidence_message(conf, pred_label)

    # Top-k predictions with probabilities
    k = len(CLASS_NAMES)
    top_indices = probs.argsort()[::-1][:k]
    
    # Create probability data for bar plot
    prob_data = pd.DataFrame({
        'Class': [CLASS_NAMES[i] for i in top_indices],
        'Probability (%)': [probs[i] * 100 for i in top_indices]
    })

    # Build result markdown
    model_display_name = model_config["display_name"]
    model_emoji = "🔷" if model_type == "cnn" else "🟢"
    
    result_md = f"""
## 🌾 Prediction Result

{model_emoji} **Model Used:** `{model_display_name}`

### Predicted Disease: `{pred_label}`
**Confidence:** {conf*100:.2f}%
"""

    return result_md, prob_data

def build_app():
    with gr.Blocks(title="Rice Leaf Disease Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🌾 Rice Leaf Disease Detection System
        ### AI-Powered Disease Classification with Dual Model Support
        
        **Available Models:**
        - 🔷 **CNN (SmallCNN)**: Accuracy {MODEL_CONFIGS['cnn_small']['accuracy']:.1%} | F1 {MODEL_CONFIGS['cnn_small']['f1']:.1%} | Fast (~15-20ms)
        - 🟢 **ViT (Small)**: Accuracy {MODEL_CONFIGS['vit_small_patch16_224']['accuracy']:.1%} | F1 {MODEL_CONFIGS['vit_small_patch16_224']['f1']:.1%} | High Accuracy (~50-100ms)
        
        > 💡 **Tip:** So sánh cả 2 models bằng cách click cả 2 buttons với cùng 1 ảnh!
        """)
        
        with gr.Tabs():
            # Tab 1: Prediction
            with gr.Tab("🔍 Predict"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload & Edit Image")
                        img_input = gr.Image(label="Rice Leaf Image", type="pil")
                        
                        # Auto color normalization toggle
                        auto_normalize = gr.Checkbox(
                            value=True, 
                            label="🎨 Auto Color Normalization (Recommended)",
                            info="Automatically adjust colors to standard green hue for better accuracy"
                        )

                        with gr.Accordion("🎨 Manual Image Adjustments (Optional)", open=False):
                            brightness = gr.Slider(0, 100, value=50, step=1,
                                                   label="Brightness")
                            contrast = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                                 label="Contrast")
                            angle = gr.Dropdown(
                                choices=[0, 90, 180, 270],
                                value=0,
                                label="Rotation"
                            )
                            flip_h = gr.Checkbox(value=False, label="Flip Horizontal")
                            flip_v = gr.Checkbox(value=False, label="Flip Vertical")

                        gr.Markdown("### 🎯 Chọn Model để Dự đoán")
                        with gr.Row():
                            btn_cnn = gr.Button(
                                "🔷 Predict with CNN",
                                variant="primary",
                                size="lg",
                                scale=1
                            )
                            btn_vit = gr.Button(
                                "🟢 Predict with ViT",
                                variant="secondary",
                                size="lg",
                                scale=1
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Results")
                        orig_img = gr.Image(label="Original Image")
                        preview_img = gr.Image(label="Edited Preview")
                        output_text = gr.Markdown(label="Prediction Results")
                        prob_plot = gr.BarPlot(
                            x="Class",
                            y="Probability (%)",
                            title="Class Probability Distribution",
                            tooltip=["Class", "Probability (%)"],
                            height=300,
                            width=500
                        )
            
            # Tab 2: Model Info
            with gr.Tab("📖 About Model"):
                gr.Markdown(get_model_info())
            
            # Tab 3: Pipeline Visualization
            with gr.Tab("🔄 Pipeline"):
                gr.Markdown(get_pipeline_tab_content())
            
            # Tab 4: Help
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## How to Use
                
                1. **Upload Image**: Click on the upload area and select a rice leaf image
                2. **Optional Editing**: Adjust brightness, contrast, etc. if needed
                3. **Predict**: Click the "Predict Disease" button
                4. **Review Results**: Check the prediction and confidence level
                
                ## Tips for Best Results
                
                - ✅ Use clear, well-lit photos of rice leaves
                - ✅ Ensure the leaf fills most of the frame
                - ✅ Avoid blurry or low-quality images
                - ⚠️ If confidence is low (<60%), the image may not be a rice leaf
                
                ## Understanding Confidence Levels
                
                - 🟢 **High (>80%)**: Model is very confident in the prediction
                - 🟡 **Medium (60-80%)**: Model is moderately confident
                - 🔴 **Low (<60%)**: Image may not be a rice leaf or quality is poor
                
                ## Disease Classes
                
                This model can detect the following rice diseases:
                """
                + "\n".join([f"- {cls}" for cls in CLASS_NAMES])
                )

                # Input controls list (HSV removed from UI)
                controls = [
                    img_input,
                    brightness,
                    contrast,
                    angle,
                    flip_h,
                    flip_v,
                    auto_normalize,
                ]

                # Display original image
                img_input.change(
                    fn=lambda x: x,
                    inputs=img_input,
                    outputs=orig_img,
                )

                # Update preview on any control change
                for c in controls:
                    c.change(
                        fn=lambda img, b, c, a, fh, fv, an: edit_image(img, b, c, 0, 1.0, 1.0, a, fh, fv, an),
                        inputs=controls,
                        outputs=preview_img,
                    )

                # CNN Button - predict with CNN model
                btn_cnn.click(
                    fn=lambda img, b, c, a, fh, fv, an: predict_from_controls(img, b, c, 0, 1.0, 1.0, a, fh, fv, an, "cnn"),
                    inputs=controls,
                    outputs=[output_text, prob_plot],
                )
                
                # ViT Button - predict with ViT model
                btn_vit.click(
                    fn=lambda img, b, c, a, fh, fv, an: predict_from_controls(img, b, c, 0, 1.0, 1.0, a, fh, fv, an, "vit"),
                    inputs=controls,
                    outputs=[output_text, prob_plot],
                )

        return demo


app = build_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", debug=True)

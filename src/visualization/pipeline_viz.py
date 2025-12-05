"""
Real-time Pipeline Visualization for Rice Leaf Disease Detection

Provides a live visualization of the processing pipeline showing:
1. Input Image
2. Preprocessing Steps
3. Model Inference
4. Prediction Output

This can be integrated into the Gradio app to show users the processing flow.
"""

import gradio as gr
import time
from typing import Tuple


def create_pipeline_diagram() -> str:
    """
    Create a Mermaid diagram showing the processing pipeline.
    
    Returns:
        Mermaid diagram markup as string
    """
    diagram = """
```mermaid
flowchart LR
    A[📤 Upload Image] --> B[🎨 Image Editing]
    B --> C[📏 Resize to 224x224]
    C --> D[🔢 Normalize]
    D --> E[🤖 ViT Model]
    E --> F[📊 Softmax]
    F --> G[✅ Prediction]
    G --> H{Confidence Check}
    H -->|High| I[🟢 Valid Result]
    H -->|Low| J[🔴 Warning]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#c5e1a5
    style F fill:#c5e1a5
    style G fill:#ffccbc
    style H fill:#ffe082
    style I fill:#a5d6a7
    style J fill:#ef9a9a
```
"""
    return diagram


def create_step_by_step_guide() -> str:
    """
    Create a detailed step-by-step processing guide.
    
    Returns:
        Markdown formatted processing steps
    """
    guide = """
## 🔄 Processing Pipeline

### Step 1: Image Upload
- User uploads a rice leaf image
- Image format: JPG, PNG, or other common formats
- Recommended: Clear, well-lit photos

### Step 2: Image Enhancement (Optional)
- **Brightness**: Adjust lighting
- **Contrast**: Enhance details
- **Hue/Saturation/Value**: Color adjustments
- **Rotation/Flip**: Orientation correction

### Step 3: Preprocessing
```python
# Resize to model input size
image = resize(image, (224, 224))

# Convert to tensor and normalize
tensor = to_tensor(image)
normalized = normalize(tensor, 
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
```

### Step 4: Model Inference
- **Model**: Vision Transformer (ViT-Small)
- **Architecture**: 16x16 patch embedding
- **Layers**: 12 transformer blocks
- **Processing**: Self-attention mechanism analyzes image patches

### Step 5: Prediction
```python
# Get model output
logits = model(normalized_image)

# Apply softmax for probabilities
probabilities = softmax(logits)

# Get top prediction
predicted_class = argmax(probabilities)
```

### Step 6: Confidence Validation
- **High Confidence (>80%)**: 🟢 Reliable prediction
- **Medium Confidence (60-80%)**: 🟡 Moderate confidence
- **Low Confidence (<60%)**: 🔴 Possibly not a rice leaf

### Step 7: Display Results
- Predicted disease class
- Confidence percentage
- Probability distribution across all classes
- Visual probability bar chart
"""
    return guide


def get_pipeline_tab_content() -> str:
    """
    Get complete content for pipeline visualization tab.
    
    Returns:
        Markdown content combining diagram and guide
    """
    content = f"""
# 🔄 Processing Pipeline Visualization

{create_pipeline_diagram()}

{create_step_by_step_guide()}

---

## ⏱️ Performance Metrics

| Stage | Typical Time |
|-------|-------------|
| Image Upload | < 1s |
| Preprocessing | < 100ms |
| Model Inference | 50-200ms (CPU) / 10-30ms (GPU) |
| Post-processing | < 50ms |
| **Total** | **< 500ms** |

---

## 🎯 Model Architecture Details

### Vision Transformer (ViT-Small)

**Patch Embedding:**
- Input: 224×224×3 RGB image
- Patches: 16×16 pixels each
- Grid: 14×14 = 196 patches
- Embedding dimension: 384

**Transformer Encoder:**
- Number of layers: 12
- Attention heads: 6
- MLP dimension: 1536
- Total parameters: ~22M

**Classification Head:**
- Global average pooling
- Linear layer: 384 → N classes (5 rice disease classes)
- Softmax activation

---

## 💡 Tips for Best Results

1. **Image Quality**: Use high-resolution, clear images
2. **Lighting**: Ensure good, even lighting
3. **Framing**: Fill frame with the rice leaf
4. **Focus**: Avoid blurry images
5. **Background**: Plain backgrounds work best
"""
    return content


# For integration into build_app():
def add_pipeline_tab():
    """
    Example integration into Gradio app.
    Usage in build_app():
    
    with gr.Tab("🔄 Pipeline"):
        gr.Markdown(get_pipeline_tab_content())
    """
    pass


if __name__ == "__main__":
   # Test the output
    print(get_pipeline_tab_content())

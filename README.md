# rice-leaf-health

Two-week, single-codebase project for rice leaf disease recognition:
- A (Course 1): Classification (ViT/ConvNeXt), Grad-CAM/SAM-based explanation, ONNX export.
- B (Course 2): Segmentation (SegFormer-B0), % diseased area, Streamlit dashboard.

## Quickstart

```bash
conda create -n rice python=3.10 -y && conda activate rice
pip install -r requirements.txt

# Train classification
python src/train.py --task cls --config configs/cls_vit_s.yaml

# Train segmentation
python src/train.py --task seg --config configs/seg_segformer_b0.yaml

# Inference (examples)
python src/infer.py --task cls --weights exports/weights/cls_vit_s_best.pt --image demo.jpg
python src/infer.py --task seg --weights exports/weights/segformer_b0_best.pt --image demo.jpg

# Apps
python src/web/app_gradio.py
streamlit run src/web/dashboard_streamlit/app.py

# API
uvicorn src.api_fastapi:app --host 0.0.0.0 --port 8000
```

. .\.venv\Scripts\Activate.ps1

## Data Layout

```
data/
├─ raw/          # raw mobile photos/videos
├─ public/       # public datasets
├─ processed/    # cleaned/resized
└─ splits/       # train/val/test txt or json
```

.\.venv\Scripts\activate.bat

python -c "import torch; print(torch.__version__)"

python -m src.tools.infer_one `
  --ckpt runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt `
  --model_name vit_small_patch16_224 `
  --img temp/test1.jpg `
  --labels_file data/splits/labels.txt `
  --img_size 224 `
  --topk 4


## KPIs
- A: F1 macro ≥ 0.80 (field-based split), CPU latency < 80ms/image (ONNXRuntime)
- B: mIoU ≥ 0.55–0.60; MAE of %area ≤ 8–10%


## 🔧 Upgrades (NN + CV + ML) — code-only
- AMP/TF32 + Gradient Accumulation + Mixup/Cutmix + EarlyStop (`src/train.py`).
- Custom CNN baseline: `src/models/cnn_small.py`.
- Grad-CAM: `python -m src.tools.gradcam --ckpt <best.pt> --model_name <cnn_small|resnet18|vit_small_patch16_224> --img <img> --img_size 224 --out out.png`
- Evaluation script: `src/tools/eval_cls.py` → classification report + confusion matrix + CSV.
- Traditional ML baseline (HSV hist + GLCM) + SVM: `src/tools/feats_svm.py`.

localhost:7860
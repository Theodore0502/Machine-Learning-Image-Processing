# rice-leaf-health

updated: 01/12/2025

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
python src/tools/web/app_gradio.py
python -m src.tools.web.app_gradio
localhost:7860
.\.venv\Scripts\Activate.ps1

python -m src.tools.infer_one `
  --ckpt runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt `
  --model_name vit_small_patch16_224 `
  --img temp/test1.jpg `
  --labels_file data/splits/labels.txt `
  --img_size 224 `
  --topk 4

python -m src.tools.eval_cls '
  --ckpt runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt '
  --model_name vit_small_patch16_224 '
  --split_file data/splits/test_cls.txt '
  --labels_file data/splits/labels.txt '
  --img_size 224 '


python -m src.tools.eval_cls \
  --ckpt runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt \
  --model_name vit_small_patch16_224 \
  --split_file data/splits/test_cls.txt \
  --labels_file data/splits/labels.txt \
  --img_size 224

python -m src.tools.eval_cls --ckpt runs/cls_vit_s_224/weights/vit_small_patch16_224_best.pt --model_name vit_small_patch16_224 --split_file data/splits/test_cls.txt --labels_file data/splits/labels.txt --img_size 224



## KPIs
- A: F1 macro ≥ 0.80 (field-based split), CPU latency < 80ms/image (ONNXRuntime)
- B: mIoU ≥ 0.55–0.60; MAE of %area ≤ 8–10%


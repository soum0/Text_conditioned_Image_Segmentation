# Prompted Segmentation for Construction QA — Hybrid CLIPSeg + U-Net Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technologies & Tools](#technologies--tools)
- [Dataset](#dataset)
- [Project Architecture / Methodology](#project-architecture--methodology)
- [Installation & Setup](#installation--setup)
- [Usage / How to Run](#usage--how-to-run)
- [Results & Key Metrics](#results--key-metrics)
- [Visualizations & Screenshots](#visualizations--screenshots)
- [Challenges Faced & How I Solved Them](#challenges-faced--how-i-solved-them)
- [Key Insights & Learnings](#key-insights--learnings)
- [Future Roadmap / Potential Improvements](#future-roadmap--potential-improvements)
- [Skills Demonstrated](#skills-demonstrated)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Author / Contact](#author--contact)

## Overview

**What this project does (simple version):**  
Given a construction-site photo and a plain-language instruction like “segment crack” or “segment tape joint”, the system automatically generates a precise binary mask that highlights exactly the requested defect.

**How it works (high level):**  
We combine two powerful ideas:

- **CLIPSeg** — a vision-language model that understands text prompts and roughly locates the described object  
- **U-Net** — a classic segmentation network that refines the coarse region into a pixel-accurate mask

This **hybrid approach** gives the flexibility of natural language prompts while delivering the kind of precise masks needed in real construction quality assurance (QA).

**Why it matters:**  
Automated defect detection (cracks, drywall seams/tape joints) can dramatically improve safety, reduce inspection costs, speed up QA processes, and help catch issues early — especially valuable on large-scale construction sites.

## Key Features

- ✅ Text-conditioned segmentation: image + natural language prompt → precise binary mask  
- ✅ Hybrid pipeline: CLIPSeg (coarse localization) → ROI cropping → U-Net refinement  
- ✅ Two real construction datasets: concrete cracks + drywall taping/seams  
- ✅ Full reproducibility: YOLO → mask conversion, training, evaluation, visualizations  
- ✅ Quantitative comparison: IoU & Dice scores per stage + failure case analysis  
- ✅ Runs in Colab (GPU) and on Mac M2 (MPS acceleration)

## Technologies & Tools

| Component              | Tool / Library                              | Version / Note                                 |
|------------------------|---------------------------------------------|------------------------------------------------|
| Deep Learning          | PyTorch                                     | 2.x                                            |
| Vision-Language        | Hugging Face transformers / CLIPSeg         | CIDAS/clipseg-rd64-refined                     |
| Segmentation backbone  | segmentation-models-pytorch (U-Net + ResNet34) | 0.3.x                                       |
| Data processing & viz  | OpenCV, Pillow, Matplotlib, Pandas, NumPy   | —                                              |
| Environment            | Jupyter / Google Colab                      | Colab recommended for GPU                      |
| Hardware acceleration  | CUDA or Apple MPS (M2)                      | —                                              |
| Dataset platform       | Roboflow exports                            | —                                              |

## Dataset

Two datasets exported from Roboflow (YOLOv8 format → converted to binary masks):

**1. Concrete Cracks**  
- Images: ~5,164 train / 201 valid / small test set  
- Characteristics: thin, faint, high-frequency crack lines → strong class imbalance

**2. Drywall Join / Tape Seam Detection**  
- Images: ~820 train / ~202 valid / small test  
- Characteristics: larger, more consistent tape/seam regions

Both datasets were annotated in YOLO polygon/bbox style → pipeline includes conversion to binary PNG masks for supervised segmentation training.

## Project Architecture / Methodology

**Simple analogy**  
CLIPSeg acts like a text-guided spotlight — it highlights roughly where “crack” or “tape joint” appears.  
U-Net acts like a fine-tipped brush that cleans up and precisely draws the edges — but needs to be told *where* to look.  
→ We let CLIPSeg find the region, crop around it, and let U-Net refine only that area.

**Detailed pipeline stages**

1. **Data preparation**  
   YOLO annotations → binary segmentation masks (PNG)

2. **Stage 1 – Supervised U-Net baseline**  
   Full-image training with ResNet34 encoder

3. **Stage 2 – CLIPSeg zero-shot**  
   Prompt → coarse heatmap (no training)

4. **Stage 3 – CLIPSeg lightweight fine-tuning**  
   Freeze image & text encoder → train segmentation head only

5. **Stage 4 – Hybrid inference (main contribution)**  
   - CLIPSeg → coarse mask  
   - Threshold + connected components → ROI bounding boxes  
   - Crop → resize → U-Net inference → paste refined mask back  
   - Merge overlapping regions + morphological cleanup

6. **Evaluation**  
   Per-image and average IoU / Dice  
   Visual overlays + CSV export of all results

## Installation & Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -U pip
pip install -r requirements.txt

## Hardware Notes

- **Training**: Google Colab GPU (T4 / P100) is strongly recommended  
- **Inference & small-scale training**: Mac M2 (Apple Silicon) works well using PyTorch MPS acceleration

## Usage / How to Run

All logic and experiments are contained in a single Jupyter notebook: **`Origin_Assignment.ipynb`**

**Recommended execution order:**

1. Run the data unzip and YOLO-to-mask conversion cells  
2. (Optional) Train the U-Net from scratch or load a pre-trained checkpoint  
3. Run CLIPSeg zero-shot inference and (optional) lightweight fine-tuning cells  
4. Execute the full hybrid pipeline cells (CLIPSeg → ROI → U-Net refinement)  
5. Generate visualizations, comparison overlays, and summary CSV files

## Results & Key Metrics

| Method                              | Crack IoU | Crack Dice | Drywall IoU | Drywall Dice |
|-------------------------------------|-----------|------------|-------------|--------------|
| U-Net baseline (fully supervised)   | 0.392     | 0.560      | —           | —            |
| CLIPSeg zero-shot                   | 0.174     | 0.284      | 0.304       | 0.435        |
| CLIPSeg fine-tuned                  | 0.372     | 0.537      | —           | —            |
| **Hybrid (CLIPSeg → U-Net refine)** | **0.361** | **0.516**  | **0.749**   | **0.844**    |

**Main observations**

- The hybrid approach delivers **very substantial gains** on drywall tape joints / seams (IoU 0.30 → 0.75)  
- Thin, faint cracks remain inherently difficult — the hybrid method still clearly outperforms zero-shot CLIPSeg, but does not surpass a fully supervised U-Net baseline


## Challenges Faced & How I Solved Them

- **Thin cracks are hard to localize**  
  → Used ROI cropping from CLIPSeg + U-Net refinement on cropped regions  
- **Severe class imbalance** (very small mask area)  
  → Applied positive class weighting, combined BCE + Dice loss, and oversampling of positive examples  
- **Roboflow ZIP download / corruption issues**  
  → Switched to Roboflow Python SDK direct download inside Colab  
- **Cross-platform compatibility (CUDA vs MPS vs CPU)**  
  → Implemented dynamic device selection: `cuda` → `mps` → `cpu`

## Key Insights & Learnings

- Combining semantic localization (CLIP-based) with precise pixel-level refinement (U-Net) is a powerful and modular design pattern  
- Thin, low-contrast structures remain one of the biggest challenges for zero-shot and few-shot vision-language segmentation models  
- Small, focused experiments (ROI-aware fine-tuning, prompt ensembles, crop-based inference) often deliver outsized performance improvements

## Future Roadmap / Potential Improvements

- Fine-tune CLIPSeg directly on ROI crops (scale-matched training-inference)  
- Prompt ensembling and paraphrasing for improved robustness  
- Train a lightweight mask decoder on frozen CLIP image features  
- Generate synthetic thin-structure (crack-like) data for better supervision  
- Model quantization and ONNX export for faster edge inference  
- Build a minimal Flask / FastAPI demo API for real-time usage

## Skills Demonstrated

- PyTorch development, vision-language models, semantic segmentation architectures  
- Hybrid model design and systematic ablation studies  
- Data preprocessing pipelines (YOLO polygon → binary mask conversion)  
- Quantitative evaluation (IoU, Dice, per-image analysis, failure-mode review)  
- Reproducible experimentation, checkpointing, logging, and result visualization  
- Clear technical writing and visual communication of ML results

## Contributing

Feel free to open issues or submit pull requests.  
Preferred style: small, focused changes with clear motivation and description.

## License

MIT License — see the [`LICENSE`](LICENSE) file for details.

## Acknowledgements

- Roboflow — convenient dataset export and management tools  
- Hugging Face — transformers library and CLIPSeg model checkpoints  
- segmentation-models-pytorch authors — excellent U-Net implementation  
- PyTorch team and the open-source machine learning community

## Author / Contact

**Soumya Singh**  
Applied Geophysics × Machine Learning

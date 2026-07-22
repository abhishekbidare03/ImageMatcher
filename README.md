# Image Matching System — DINOv2 + Cosine Similarity

A lightweight, **fully offline** image matching pipeline that uses
**DINOv2 Small** (`dinov2_vits14`) embeddings and cosine similarity to find
the best matching captured image for every seed image.

---

## Project Structure

```
project/
├── seed_images/        ← place your reference images here
├── captured_images/    ← place your drone/captured images here
├── results/
│   ├── matches.json    ← generated automatically
│   └── report.csv      ← generated automatically
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Jetson Orin Nano note**: Install the Jetson-optimised PyTorch wheel first
> (from NVIDIA JetPack / Jetson Zoo), then install the remaining packages with pip.

### 2 — Download DINOv2 weights (first run only)

The first execution calls `torch.hub.load("facebookresearch/dinov2", …)` which
downloads the `dinov2_vits14` weights (~85 MB) and caches them in
`~/.cache/torch/hub/`.  All subsequent runs are **fully offline**.

---

## Usage

```bash
# 1. Place reference images in seed_images/
# 2. Place captured images in captured_images/
# 3. Run:
python main.py
```

---

## Processing Pipeline

| Step | Action |
|------|--------|
| 1 | Load all seed images from `seed_images/` |
| 2 | Load all captured images from `captured_images/` |
| 3 | Resize every image to **128 × 128 px** via OpenCV |
| 4 | Extract **DINOv2 Small** CLS-token embeddings |
| 5 | Compute **cosine similarity** (seed × captured matrix) |
| 6 | For each seed, select the captured image with the highest score |
| 7 | Write `results/matches.json` and `results/report.csv` |

---

## Output

### `results/matches.json`
```json
{
  "seed1.jpg": { "best_match": "image4.jpg", "similarity": 0.91 },
  "seed2.jpg": { "best_match": "image7.jpg", "similarity": 0.88 }
}
```

### `results/report.csv`
```
Seed Image,Matched Image,Similarity Score
seed1.jpg,image4.jpg,0.91
seed2.jpg,image7.jpg,0.88
```

---

## Supported Image Formats

`.jpg` / `.jpeg` / `.png` / `.bmp` / `.tiff` / `.tif` / `.webp`

---

## Hardware

- Designed to run on **NVIDIA Jetson Orin Nano** (CUDA supported)
- Falls back to CPU automatically if CUDA is unavailable
- No internet connection required after the initial weight download
- No training, fine-tuning, or labelling required

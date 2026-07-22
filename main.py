"""
Patch-Based Geometric Matching — ORB + RANSAC (Primary)
Optional DINOv2 verification (secondary)
=========================================================
Pipeline:
  1.  Load seed -> grayscale -> ORB keypoints + descriptors.
  2.  For every test image, extract overlapping patches (3 scales).
  3.  For every patch: ORB keypoints + BFMatcher + Lowe ratio test.
  4.  RANSAC homography -> total_matches, inlier_count, inlier_ratio.
  5.  FinalScore = 0.6 x InlierRatio + 0.4 x MatchQuality.
  6.  Reject patches with InlierRatio < 0.25.
  7.  [OPTIONAL] DINOv2 similarity appended to top-K for reference.
  8.  Save annotated JPEG + matches.json + report.csv.

Usage:
    python main.py
"""

import sys
import json
import csv
import pathlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = pathlib.Path(__file__).parent.resolve()
SEED_DIR     = BASE_DIR / "seed_images"
TEST_DIR     = BASE_DIR / "test"
RESULTS_DIR  = BASE_DIR / "results"
MATCHES_JSON = RESULTS_DIR / "matches.json"
REPORT_CSV   = RESULTS_DIR / "report.csv"

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PATCH_CONFIGS = [
    (128,  64),     # (patch_size_px, stride_px) -- 50% overlap
    (192,  96),
    (256, 128),
]

ORB_MAX_KP            = 2000    # ORB keypoint limit per image/patch
ORB_RATIO             = 0.75    # Lowe ratio test threshold
RANSAC_THRESH         = 4.0     # RANSAC reprojection threshold (px)
MIN_MATCHES_HOMOG     = 6       # minimum good matches for homography
                                # (4 = absolute minimum; 6+ avoids degenerate all-inlier cases)
INLIER_RATIO_MIN      = 0.0     # 0.0 = no hard rejection; every patch gets an ORB score
                                # Raise this (e.g. 0.4) only if images have strong corners/edges
MATCH_QUALITY_CAP     = 30      # inlier count that maps to MatchQuality = 1.0

SCORE_INLIER_W        = 0.6
SCORE_QUALITY_W       = 0.4

TOP_CANDIDATES        = 20      # patches passed to optional DINO stage
TOP_OUTPUT_K          = 10      # patches in final output

# ---------------------------------------------------------------------------
# DINOv2 OPTIONAL -- set True to enable secondary DINO scoring
# ---------------------------------------------------------------------------
ENABLE_DINO      = True
DINO_TOP_K       = 10           # how many top ORB candidates to verify
DINO_BLEND_W     = 0.2          # weight of DINO score in blended final score
# When DINO is enabled:
#   BlendedScore = (1 - DINO_BLEND_W) x ORBFinalScore + DINO_BLEND_W x DinoScore

RESIZE_DIM       = 128          # seed resize for ORB + DINO input
DINO_CENTRE_CROP = 126          # 9 x 14, nearest multiple of 14 below 128
DINO_EMBED_BATCH = 32

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class PatchResult:
    x:             int
    y:             int
    w:             int
    h:             int
    total_matches: int   = 0
    inlier_count:  int   = 0
    inlier_ratio:  float = 0.0
    match_quality: float = 0.0
    orb_score:     float = 0.0   # = 0.6*inlier_ratio + 0.4*match_quality
    dino_score:    float = 0.0   # optional
    final_score:   float = 0.0   # blended if DINO enabled, else = orb_score


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    for d in (SEED_DIR, TEST_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def list_images(folder: pathlib.Path) -> list:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def load_image(path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    return img


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Patch generation
# ---------------------------------------------------------------------------

def generate_patches(img: np.ndarray, patch_size: int, stride: int) -> list:
    """
    Sliding-window extraction with given stride (50% overlap).
    Right and bottom edges always included.
    Returns list of (x, y, patch_bgr).
    """
    h, w = img.shape[:2]
    if patch_size > h or patch_size > w:
        return []

    def grid(length: int) -> list:
        pts = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if pts[-1] != last:
            pts.append(last)
        return pts

    seen    = set()
    patches = []
    for y in grid(h):
        for x in grid(w):
            if (x, y) not in seen:
                seen.add((x, y))
                patches.append((x, y, img[y:y + patch_size, x:x + patch_size]))
    return patches


# ---------------------------------------------------------------------------
# ORB + RANSAC geometric scoring
# ---------------------------------------------------------------------------

def orb_ransac_score(seed_gray:   np.ndarray,
                     seed_kp:     list,
                     seed_des:    np.ndarray,
                     patch_bgr:   np.ndarray) -> PatchResult:
    """
    Match seed ORB descriptors against a single patch using BFMatcher + RANSAC.
    Returns a PatchResult (x,y,w,h all 0 -- caller fills geometry).
    """
    orb        = cv2.ORB_create(nfeatures=ORB_MAX_KP)
    patch_gray = to_gray(resize_to(patch_bgr, RESIZE_DIM))

    kp2, des2  = orb.detectAndCompute(patch_gray, None)

    res = PatchResult(x=0, y=0, w=0, h=0)

    if des2 is None or len(kp2) < 2 or seed_des is None:
        return res

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw = matcher.knnMatch(seed_des, des2, k=2)
    except cv2.error:
        return res

    # Lowe ratio test
    good = [m for m, n in raw if m.distance < ORB_RATIO * n.distance]
    res.total_matches = len(good)

    if len(good) < MIN_MATCHES_HOMOG:
        return res

    # RANSAC homography
    src_pts = np.float32([seed_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt     for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts,
                                 cv2.RANSAC, RANSAC_THRESH)

    if mask is None:
        return res

    inliers            = int(mask.sum())
    res.inlier_count   = inliers
    res.inlier_ratio   = round(inliers / max(len(good), 1), 4)
    res.match_quality  = round(min(inliers / MATCH_QUALITY_CAP, 1.0), 4)
    res.orb_score      = round(
        SCORE_INLIER_W * res.inlier_ratio +
        SCORE_QUALITY_W * res.match_quality, 4
    )
    return res


# ---------------------------------------------------------------------------
# Optional DINOv2 verification
# ---------------------------------------------------------------------------

_dino_model     = None
_dino_transform = None


def _load_dino(device):
    global _dino_model, _dino_transform
    if _dino_model is not None:
        return _dino_model, _dino_transform

    import torch
    import torchvision.transforms as T

    print("[DINO] Loading DINOv2 Small (dinov2_vits14) ...")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
    )
    model.eval().to(device)

    transform = T.Compose([
        T.Resize(RESIZE_DIM),
        T.CenterCrop(DINO_CENTRE_CROP),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    _dino_model     = model
    _dino_transform = transform
    print(f"[DINO] Model ready on {device}.")
    return model, transform


def dino_embed(bgr_images: list, device) -> "np.ndarray":
    import torch
    from PIL import Image as PILImage

    model, transform = _load_dino(device)

    def to_pil(bgr):
        return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    tensors = [transform(to_pil(img)) for img in bgr_images]
    parts   = []
    with torch.no_grad():
        for start in range(0, len(tensors), DINO_EMBED_BATCH):
            batch = torch.stack(tensors[start:start + DINO_EMBED_BATCH]).to(device)
            feats = model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            parts.append(feats.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Per-image full pipeline
# ---------------------------------------------------------------------------

def process_image(seed_kp:      list,
                  seed_des:     np.ndarray,
                  seed_gray:    np.ndarray,
                  seed_bgr:     np.ndarray,
                  cap_img:      np.ndarray,
                  cap_name:     str,
                  device) -> list:
    """
    Run full geometric + optional DINO pipeline on one test image.
    Returns up to TOP_OUTPUT_K PatchResult objects, sorted best first.
    """
    all_candidates = []

    # ------------------------------------------------------------------
    # Stage 1: ORB + RANSAC on all patches
    # ------------------------------------------------------------------
    for patch_size, stride in PATCH_CONFIGS:
        patches = generate_patches(cap_img, patch_size, stride)
        if not patches:
            print(f"    [SKIP] {patch_size}px patch -- image too small")
            continue

        print(f"    [ORB]  scale={patch_size:3d}px  stride={stride:3d}px  "
              f"patches={len(patches):4d}  ...", end="", flush=True)

        accepted = 0
        for x, y, patch_bgr in patches:
            res   = orb_ransac_score(seed_gray, seed_kp, seed_des, patch_bgr)
            res.x = x
            res.y = y
            res.w = patch_size
            res.h = patch_size

            # Reject low-inlier patches
            if res.inlier_ratio < INLIER_RATIO_MIN:
                continue

            res.final_score = res.orb_score    # will be blended if DINO runs
            all_candidates.append(res)
            accepted += 1

        print(f"  accepted={accepted}")

    if not all_candidates:
        print("    [WARN] No patches passed the inlier-ratio threshold "
              f"(>= {INLIER_RATIO_MIN:.2f}).")
        return []

    # Sort by ORB score descending
    all_candidates.sort(key=lambda r: r.orb_score, reverse=True)
    top_candidates = all_candidates[:TOP_CANDIDATES]

    # ------------------------------------------------------------------
    # Stage 2 (optional): DINOv2 verification on top-K
    # ------------------------------------------------------------------
    if ENABLE_DINO and top_candidates:
        import torch
        device_obj = torch.device(device if isinstance(device, str) else device)

        k = min(DINO_TOP_K, len(top_candidates))
        dino_batch = top_candidates[:k]
        print(f"    [DINO] Verifying top {k} candidates ...", end="", flush=True)

        patch_imgs = [
            cap_img[r.y:r.y + r.h, r.x:r.x + r.w]
            for r in dino_batch
        ]
        seed_emb  = dino_embed([seed_bgr], device_obj)          # (1, D)
        patch_emb = dino_embed(patch_imgs, device_obj)          # (k, D)
        dino_sims = (seed_emb @ patch_emb.T).flatten()          # (k,)

        for i, res in enumerate(dino_batch):
            res.dino_score  = round(float(dino_sims[i]), 4)
            res.final_score = round(
                (1 - DINO_BLEND_W) * res.orb_score +
                DINO_BLEND_W * res.dino_score, 4
            )

        print(f"  max_dino={dino_sims.max():.4f}")

        # Re-sort by blended score
        top_candidates.sort(key=lambda r: r.final_score, reverse=True)

    return top_candidates[:TOP_OUTPUT_K]


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

def save_visual(cap_img:  np.ndarray,
                patches:  list,
                cap_name: str) -> None:
    out  = cap_img.copy()
    best = patches[0]

    # Secondary patches -- green, thin
    for res in patches[1:]:
        cv2.rectangle(out,
                      (res.x, res.y),
                      (res.x + res.w, res.y + res.h),
                      (0, 200, 0), 1)

    # Best patch -- cyan, thick
    cv2.rectangle(out,
                  (best.x, best.y),
                  (best.x + best.w, best.y + best.h),
                  (255, 255, 0), 3)

    # Text overlay
    lines = [
        "BEST MATCH",
        f"Final  : {best.final_score:.4f}",
        f"Inlier : {best.inlier_ratio:.4f}",
        f"Quality: {best.match_quality:.4f}",
        f"Matches: {best.total_matches}  Inliers: {best.inlier_count}",
        f"Loc    : ({best.x},{best.y})  {best.w}x{best.h}",
    ]
    if ENABLE_DINO and best.dino_score > 0:
        lines.insert(2, f"DINO   : {best.dino_score:.4f}")

    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.55
    thickness = 1
    pad       = 6
    line_h    = 20

    text_w = max(cv2.getTextSize(l, font, fscale, thickness)[0][0]
                 for l in lines) + pad * 2
    text_h = line_h * len(lines) + pad * 2

    lx = max(best.x, 0)
    ly = max(best.y - text_h - 4, 0)
    if lx + text_w > out.shape[1]:
        lx = max(out.shape[1] - text_w, 0)

    cv2.rectangle(out, (lx, ly), (lx + text_w, ly + text_h), (20, 20, 20), -1)
    cv2.rectangle(out, (lx, ly), (lx + text_w, ly + text_h), (255, 255, 0),  1)

    for i, line in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (200, 200, 200)
        cv2.putText(out, line,
                    (lx + pad, ly + pad + (i + 1) * line_h - 2),
                    font, fscale, color, thickness, cv2.LINE_AA)

    stem     = pathlib.Path(cap_name).stem
    out_path = RESULTS_DIR / f"{stem}_result.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"    [VIZ]  Saved -> {out_path.name}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def log_summary(seed_name: str, cap_name: str, patches: list) -> None:
    div = "-" * 56
    print(f"\n{div}")
    print(f"  Seed    : {seed_name}")
    print(f"  Captured: {cap_name}")
    print(div)
    for i, res in enumerate(patches, 1):
        dino_line = (f"    DINO     : {res.dino_score:.4f}\n"
                     if ENABLE_DINO and res.dino_score > 0 else "")
        print(
            f"\n  Rank {i}\n"
            f"    Location : ({res.x}, {res.y})\n"
            f"    Patch    : {res.w}x{res.h}\n"
            f"    Matches  : {res.total_matches}  Inliers: {res.inlier_count}\n"
            f"    InlierR  : {res.inlier_ratio:.4f}\n"
            f"    Quality  : {res.match_quality:.4f}\n"
            f"    ORBScore : {res.orb_score:.4f}\n"
            f"{dino_line}"
            f"    Final    : {res.final_score:.4f}"
        )
    print(div)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    seed_paths = list_images(SEED_DIR)
    test_paths = list_images(TEST_DIR)

    if not seed_paths:
        sys.exit(f"[ERROR] No images in '{SEED_DIR}'.")
    if not test_paths:
        sys.exit(f"[ERROR] No images in '{TEST_DIR}'.")

    seed_path = seed_paths[0]
    seed_name = seed_path.name
    print(f"\n[INFO] Seed image  : '{seed_name}'  ({len(seed_paths)} available, using first)")
    print(f"[INFO] Test images : {len(test_paths)}")
    print(f"[INFO] Patch scales: {[c[0] for c in PATCH_CONFIGS]} px  (50% stride)")
    print(f"[INFO] Scoring     : {SCORE_INLIER_W} x InlierRatio"
          f" + {SCORE_QUALITY_W} x MatchQuality")
    print(f"[INFO] Min inlier ratio threshold : {INLIER_RATIO_MIN}")
    print(f"[INFO] DINO verification : {'ENABLED (blend w=' + str(DINO_BLEND_W) + ')' if ENABLE_DINO else 'DISABLED'}")

    # Load seed
    seed_bgr  = load_image(seed_path)
    seed_bgr  = resize_to(seed_bgr, RESIZE_DIM)    # 128x128 for ORB
    seed_gray = to_gray(seed_bgr)

    print(f"\n[INFO] Extracting ORB features from seed '{seed_name}' ...")
    orb_detector = cv2.ORB_create(nfeatures=ORB_MAX_KP)
    seed_kp, seed_des = orb_detector.detectAndCompute(seed_gray, None)

    if seed_des is None or len(seed_kp) < 4:
        sys.exit(f"[ERROR] Not enough ORB keypoints in seed image "
                 f"(found {len(seed_kp) if seed_kp else 0}). "
                 "Try a more textured seed image.")

    print(f"[INFO] Seed keypoints: {len(seed_kp)}")

    # Device (for optional DINO)
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"
    print(f"[INFO] Device      : {device}\n")

    all_output: dict = {}

    for test_path in test_paths:
        cap_name = test_path.name
        print(f"\n{'=' * 60}")
        print(f"[INFO] Processing test image: '{cap_name}'")
        print(f"{'=' * 60}")

        try:
            cap_img = load_image(test_path)
        except RuntimeError as exc:
            print(f"  [WARN] Skipping -- {exc}")
            continue

        h, w = cap_img.shape[:2]
        print(f"    Image dimensions: {w}x{h} px")

        top_patches = process_image(
            seed_kp, seed_des, seed_gray, seed_bgr,
            cap_img, cap_name, device
        )

        if not top_patches:
            print(f"  [RESULT] No valid matches found in '{cap_name}'.")
            all_output[cap_name] = []
            continue

        log_summary(seed_name, cap_name, top_patches)
        save_visual(cap_img, top_patches, cap_name)
        all_output[cap_name] = [asdict(p) for p in top_patches]

    # ------------------------------------------------------------------
    # Save matches.json
    # ------------------------------------------------------------------
    json_out = {
        "seed": seed_name,
        "config": {
            "patch_scales":          [c[0] for c in PATCH_CONFIGS],
            "strides":               [c[1] for c in PATCH_CONFIGS],
            "orb_max_keypoints":     ORB_MAX_KP,
            "lowe_ratio":            ORB_RATIO,
            "ransac_threshold":      RANSAC_THRESH,
            "min_inlier_ratio":      INLIER_RATIO_MIN,
            "score_inlier_weight":   SCORE_INLIER_W,
            "score_quality_weight":  SCORE_QUALITY_W,
            "dino_enabled":          ENABLE_DINO,
            "dino_blend_weight":     DINO_BLEND_W if ENABLE_DINO else 0,
        },
        "results": all_output,
    }
    with open(MATCHES_JSON, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[INFO] Saved -> {MATCHES_JSON}")

    # ------------------------------------------------------------------
    # Save report.csv
    # ------------------------------------------------------------------
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Seed Image", "Test Image", "Rank",
            "X", "Y", "Width", "Height",
            "Total Matches", "Inlier Count", "Inlier Ratio",
            "Match Quality", "ORB Score", "DINO Score", "Final Score",
        ])
        for cap_name, patches in all_output.items():
            for rank, p in enumerate(patches, 1):
                writer.writerow([
                    seed_name, cap_name, rank,
                    p["x"], p["y"], p["w"], p["h"],
                    p["total_matches"], p["inlier_count"], p["inlier_ratio"],
                    p["match_quality"], p["orb_score"],
                    p["dino_score"], p["final_score"],
                ])
    print(f"[INFO] Saved -> {REPORT_CSV}")
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()

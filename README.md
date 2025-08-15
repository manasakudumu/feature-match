# Feature Matching (Harris + SIFT-like) — Minimal OpenCV Version

Implements a simple local feature-matching pipeline for comparing two images:

- **Detection:** Harris corner detection with dilation-based non-maximum suppression (NMS)
- **Description:** Choice of raw patch descriptor or SIFT-like 4×4×8 orientation histogram (RootSIFT-style normalization)
- **Matching:** Lowe’s ratio test, optional symmetric match filtering, and RANSAC-based geometric verification for robustness
- **Evaluation:** Accuracy measurement against ground-truth correspondence masks
- **Visualization:** Side-by-side color image panel with hollow black circles for detected keypoints and green/red lines for correct/incorrect matches

---

## Technologies

- **Python 3**
- **OpenCV** (feature detection, description, and image processing)
- **NumPy** (matrix and array operations)
- **Matplotlib** (visualizations)

---

## Visualization

![Feature match](match.png)

import numpy as np
import cv2
import matplotlib.pyplot as plt

# converts image to bgr format
def _to_bgr8(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        m = float(img.max()) if img.size else 1.0
        if m <= 1.0:
            img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# displays viz of feature correspondences between two images
def show_correspondences(imgA, imgB, x1, y1, x2, y2, matches, good_matches=None, filename=None, topk=100, ring_radius=8, ring_thick=2, center_dot=2, line_thick=2):
    a = _to_bgr8(imgA)
    b = _to_bgr8(imgB)
    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = a
    canvas[:h2, w1:w1 + w2] = b
    for x, y in zip(x1, y1):
        p = (int(x), int(y))
        cv2.circle(canvas, p, ring_radius, (0, 0, 0), ring_thick)
        cv2.circle(canvas, p, center_dot, (0, 0, 0), -1)
    for x, y in zip(x2, y2):
        p = (int(x + w1), int(y))
        cv2.circle(canvas, p, ring_radius, (0, 0, 0), ring_thick)
        cv2.circle(canvas, p, center_dot, (0, 0, 0), -1)
    k = min(topk, len(matches))
    for i in range(k):
        i1, i2 = matches[i]
        p1 = (int(x1[i1]), int(y1[i1]))
        p2 = (int(x2[i2] + w1), int(y2[i2]))
        if good_matches is None:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0) if good_matches[i] else (0, 0, 255)
        cv2.line(canvas, p1, p2, color, line_thick)
        cv2.circle(canvas, p1, max(2, center_dot), color, -1)
        cv2.circle(canvas, p2, max(2, center_dot), color, -1)
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"top {k} matches")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    if filename:
        cv2.imwrite(filename, canvas)

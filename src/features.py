import numpy as np
import cv2
from typing import Tuple

# convert image to grayscale float32
def _to_gray_f32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        g = image
    g = g.astype(np.float32)
    if g.max() > 1.0:
        g /= 255.0
    return g

# remove keypoints too close to image border
def _filter_border_points(x: np.ndarray, y: np.ndarray, h: int, w: int, half: int):
    xi = x.astype(np.int32)
    yi = y.astype(np.int32)
    keep = (yi - half >= 0) & (yi + half < h) & (xi - half >= 0) & (xi + half < w)
    return xi[keep], yi[keep], keep

#compute normalized raw pixel patch descriptors
def _normalized_patch_descriptors(gray: np.ndarray, x: np.ndarray, y: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape
    half = patch_size // 2
    xi, yi, _ = _filter_border_points(x, y, h, w, half)
    descs = []
    for cx, cy in zip(xi, yi):
        patch = gray[cy - half: cy + half, cx - half: cx + half].copy()
        v = patch.reshape(-1).astype(np.float32)
        v -= v.mean()
        v /= (np.linalg.norm(v) + 1e-8)
        descs.append(v)
    if len(descs) == 0:
        return np.zeros((0, patch_size * patch_size), dtype=np.float32), xi.astype(np.float32), yi.astype(np.float32)
    return np.vstack(descs), xi.astype(np.float32), yi.astype(np.float32)

# compute sift like histogram
def _sift_like_descriptors(gray: np.ndarray, x: np.ndarray, y: np.ndarray, patch_size: int = 16, num_cells: int = 4, num_bins: int = 8, gaussian_sigma: float | None = None, orientations_deg: np.ndarray | None = None,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape
    half = patch_size // 2
    cell_size = patch_size // num_cells
    assert patch_size % num_cells == 0, "must be divisible by num of cells"
    xi, yi, keep_mask = _filter_border_points(x, y, h, w, half)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 360.0) % 360.0 
    if gaussian_sigma is None:
        gaussian_sigma = 0.5 * patch_size
    g1d = cv2.getGaussianKernel(ksize=patch_size, sigma=gaussian_sigma)
    weight = (g1d @ g1d.T).astype(np.float32)
    d = num_cells * num_cells * num_bins
    descs = np.zeros((len(xi), d), dtype=np.float32)
    bins = num_bins / 360.0
    for idx, (cx, cy) in enumerate(zip(xi, yi)):
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        m = mag[y0:y1, x0:x1] * weight
        a = ang[y0:y1, x0:x1].copy()
        if orientations_deg is not None:
            theta = float(orientations_deg[keep_mask][idx])
            a = (a - theta) % 360.0
        hist = np.zeros((num_cells, num_cells, num_bins), dtype=np.float32)
        for cy_cell in range(num_cells):
            for cx_cell in range(num_cells):
                ys = cy_cell * cell_size
                xs = cx_cell * cell_size
                mcell = m[ys:ys + cell_size, xs:xs + cell_size]
                acell = a[ys:ys + cell_size, xs:xs + cell_size]
                b = np.floor(acell * bins).astype(np.int32)
                b = np.clip(b, 0, num_bins - 1)
                for i in range(mcell.shape[0]):
                    for j in range(mcell.shape[1]):
                        hist[cy_cell, cx_cell, b[i, j]] += mcell[i, j]
        v = hist.reshape(-1)
        v = v / (np.sum(v) + 1e-8) 
        v = np.sqrt(v) 
        v = v / (np.linalg.norm(v) + 1e-8) 
        descs[idx, :] = v
    return descs, xi.astype(np.float32), yi.astype(np.float32)

#draw detected feature points as black circles
def plot_feature_points(image, xs, ys):
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for x, y in zip(xs, ys):
        p = (int(x), int(y))
        cv2.circle(vis, p, 8, (0, 0, 0), 2) 
        cv2.circle(vis, p, 2, (0, 0, 0), -1) 
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Feature points ({len(xs)})')
    plt.axis('off')
    plt.show()

# detect corners using Harris + non-maximum suppression
def get_feature_points(image, feature_width):
    gray = _to_gray_f32(image)
    h, w = gray.shape[:2]
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    Sxx = cv2.GaussianBlur(Ixx, (0, 0), 1.5)
    Syy = cv2.GaussianBlur(Iyy, (0, 0), 1.5)
    Sxy = cv2.GaussianBlur(Ixy, (0, 0), 1.5)
    r = (Sxx * Syy - Sxy * Sxy) - 0.04 * (Sxx + Syy) ** 2
    Rmax = float(r.max()) if r.size else 0.0
    if Rmax <= 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    nms_radius = max(4, feature_width // 2)
    size = 3 * nms_radius + 1
    kernel = np.ones((size, size), np.uint8)
    dil = cv2.dilate(r, kernel)
    thr = 0.005 * Rmax
    peaks = (r == dil) & (r > thr)
    ys, xs = np.where(peaks)
    half = feature_width // 2
    keep = (xs >= half) & (xs < w - half) & (ys >= half) & (ys < h - half)
    xs, ys = xs[keep].astype(np.float32), ys[keep].astype(np.float32)
    if xs.size > 2000:
        scores = r[ys.astype(int), xs.astype(int)]
        order = np.argsort(-scores)[:2000]
        xs, ys = xs[order], ys[order]
    return xs, ys

# get descriptors for each keypoint using patch or SIFT-like method
def get_feature_descriptors(image, xs, ys, feature_width, mode):
    gray = _to_gray_f32(image)
    if mode == "patch":
        feats, _, _ = _normalized_patch_descriptors(gray, xs, ys, patch_size=feature_width)
        return feats
    elif mode == "sift":
        feats, _, _ = _sift_like_descriptors(gray, xs, ys, patch_size=feature_width, num_cells=4, num_bins=8, gaussian_sigma=0.5 * feature_width, orientations_deg=None,)
        return feats
    else:
        raise ValueError("mode must be 'patch' or 'sift'")

# match descriptors between two images using lowe's ratio, mutual check, and RANSAC
def match_features(desc1, desc2, x1, y1, x2, y2, ratio_thresh=0.90, mutual=True, use_ransac=True, ransac_thresh=5.0, ransac_conf=0.99):
    if desc1.size == 0 or desc2.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_1to2 = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    good_1to2 = {}  
    for m, n in raw_1to2:
        r = m.distance / (n.distance + 1e-12)
        if r < ratio_thresh:
            good_1to2[m.queryIdx] = (m.trainIdx, r)
    pairs = []
    ratios = []
    if mutual:
        raw_2to1 = bf.knnMatch(desc2.astype(np.float32), desc1.astype(np.float32), k=2)
        good_2to1 = {}
        for m, n in raw_2to1:
            r = m.distance / (n.distance + 1e-12)
            if r < ratio_thresh:
                good_2to1[m.queryIdx] = m.trainIdx
        for i1, (i2, r) in good_1to2.items():
            if i2 in good_2to1 and good_2to1[i2] == i1:
                pairs.append((i1, i2))
                ratios.append(r)
    else:
        for i1, (i2, r) in good_1to2.items():
            pairs.append((i1, i2))
            ratios.append(r)
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    matches = np.array(pairs, dtype=np.int32)
    ratios = np.array(ratios, dtype=np.float32)
    if use_ransac and len(matches) >= 8:
        pts1 = np.float32([[x1[i], y1[i]] for i, _ in matches])
        pts2 = np.float32([[x2[j], y2[j]] for _, j in matches])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,ransacReprojThreshold=ransac_thresh, confidence=ransac_conf)
        if mask is None:
            return np.zeros((0, 2), dtype=np.int32)
        mask = mask.ravel().astype(bool)
        matches = matches[mask]
        ratios  = ratios[mask]
    order = np.argsort(ratios)
    matches = matches[order]
    return matches
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def evaluate_correspondence_mask(x1, y1, x2, y2, mat_dir: Path, factor: float = 1.0):
    x1 = x1 / factor
    y1 = y1 / factor
    x2 = x2 / factor
    y2 = y2 / factor
    gt = loadmat(str("data/NotreDame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat"))
    x1_gt, y1_gt = gt["x1"].ravel(), gt["y1"].ravel()
    x2_gt, y2_gt = gt["x2"].ravel(), gt["y2"].ravel()
    good = np.zeros(len(x1), dtype=bool)
    for i in range(len(x1)):
        dx = x1[i] - x1_gt
        dy = y1[i] - y1_gt
        d = np.sqrt(dx*dx + dy*dy)
        j = np.argmin(d)
        cur_off = np.array([x1[i]-x2[i], y1[i]-y2[i]])
        gt_off  = np.array([x1_gt[j]-x2_gt[j],   y1_gt[j]-y2_gt[j]])
        match_err = np.linalg.norm(cur_off - gt_off)
        good[i] = not (d[j] > 150 or match_err > 25)
    return good
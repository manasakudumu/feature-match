import argparse
import numpy as np
import cv2
import features
import visualize
from helpers import evaluate_correspondence_mask

def load_data(which):
    if which == "notre_dame":
        img1 = "data/NotreDame/left.jpg"
        img2 = "data/NotreDame/right.jpg"
        eval_mat = "data/NotreDame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat"
    else:
        raise ValueError("supported: notre_dame, custom")
    i1 = cv2.imread(str(img1), cv2.IMREAD_COLOR)
    i2 = cv2.imread(str(img2), cv2.IMREAD_COLOR)
    if i1 is None or i2 is None:
        raise FileNotFoundError("missing images")
    return i1, i2, eval_mat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, choices=["notre_dame", "custom"])
    ap.add_argument("-p", "--points", required=True, choices=["student_points", "cheat_points"])
    ap.add_argument("--sift", action="store_true")
    ap.add_argument("--feature_width", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()
    img1_color, img2_color, eval_mat = load_data(args.data)
    gray1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    image1 = cv2.resize(gray1, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(gray2, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
    # vis1 = cv2.resize(img1_color, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
    # vis2 = cv2.resize(img2_color, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
    if args.points == "student_points":
        x1, y1 = features.get_feature_points((image1 * 255).astype("uint8"), args.feature_width)
        x2, y2 = features.get_feature_points((image2 * 255).astype("uint8"), args.feature_width)
    else:
        if eval_mat is None:
            raise ValueError("cheat_points")
        from helpers import cheat_feature_points  
        x1, y1, x2, y2 = cheat_feature_points(eval_mat, factor=args.scale)
    # student.plot_feature_points(img1_vis, x1, y1)
    # student.plot_feature_points(img2_vis, x2, y2)
    mode = "sift" if args.sift else "patch"
    f1 = features.get_feature_descriptors(image1, x1, y1, args.feature_width, mode)
    f2 = features.get_feature_descriptors(image2, x2, y2, args.feature_width, mode)
    matches = features.match_features(f1, f2, x1, y1, x2, y2)
    print("matches:", matches.shape[0])
    K = min(args.topk, len(matches))
    good_mask = evaluate_correspondence_mask(
        x1[matches[:K, 0]], y1[matches[:K, 0]], x2[matches[:K, 1]], y2[matches[:K, 1]], "data/NotreDame",
        factor=args.scale,)
    correct = int(good_mask.sum())
    print(f"top-{K} accuracy: {100.0*correct/K:.2f}%  ({correct}/{K})")
    visualize.show_correspondences(
        cv2.resize(img1_color, None, fx=args.scale, fy=args.scale),
        cv2.resize(img2_color, None, fx=args.scale, fy=args.scale),
        x1, y1, x2, y2, matches[:K], good_matches=good_mask, filename=None, topk=K)
if __name__ == "__main__":
    main()
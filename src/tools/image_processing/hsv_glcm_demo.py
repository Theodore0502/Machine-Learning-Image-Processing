# src/tools/hsv_glcm_demo.py
import argparse, os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import graycomatrix, graycoprops

def hsv_hist(img, bins=32):
    hsv = rgb2hsv(np.asarray(img)/255.0)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    h_hist, _ = np.histogram(H, bins=bins, range=(0,1), density=True)
    s_hist, _ = np.histogram(S, bins=bins, range=(0,1), density=True)
    v_hist, _ = np.histogram(V, bins=bins, range=(0,1), density=True)
    return h_hist, s_hist, v_hist

def glcm_feats(img_gray_u8, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    glcm = graycomatrix(img_gray_u8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        feats[prop] = graycoprops(glcm, prop).mean()
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out_prefix", default="ip_demo")
    ap.add_argument("--bins", type=int, default=32)
    args = ap.parse_args()

    img = Image.open(args.img).convert("RGB")

    # 1) HSV hist
    h_hist, s_hist, v_hist = hsv_hist(img, bins=args.bins)
    x = np.arange(args.bins)
    plt.figure()
    plt.title("Histogram HSV (chuẩn hoá)")
    plt.plot(x, h_hist, label="H")
    plt.plot(x, s_hist, label="S")
    plt.plot(x, v_hist, label="V")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.out_prefix+"_hsv_hist.png", dpi=200)

    # 2) GLCM
    gray = rgb2gray(np.asarray(img))  # [0,1]
    gray_u8 = (gray*255).astype(np.uint8)
    feats = glcm_feats(gray_u8)
    plt.figure()
    plt.title("GLCM features")
    names = list(feats.keys()); vals = [feats[k] for k in names]
    plt.bar(names, vals); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_prefix+"_glcm.png", dpi=200)

    print("Saved:", args.out_prefix+"_hsv_hist.png", "and", args.out_prefix+"_glcm.png")
    print("GLCM:", feats)

if __name__ == "__main__":
    main()

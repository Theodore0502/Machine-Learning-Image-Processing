import argparse, os, numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def parse_split(fp):
    paths, labels = [], []
    with open(fp,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p, lab = line.rsplit(" ",1)
            paths.append(p); labels.append(int(lab))
    return paths, np.array(labels)
def feats(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    hsv = rgb2hsv(img/255.0)
    h = np.histogram(hsv[:,:,0], bins=16, range=(0,1), density=True)[0]
    s = np.histogram(hsv[:,:,1], bins=16, range=(0,1), density=True)[0]
    v = np.histogram(hsv[:,:,2], bins=16, range=(0,1), density=True)[0]
    gray = (rgb2gray(img)*255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1,2], angles=[0,0.785398,1.570796,2.356194],
                        levels=256, symmetric=True, normed=True)
    props = []
    for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        try:
            props.append(graycoprops(glcm, prop).mean())
        except Exception:
            props.append(0.0)
    return np.concatenate([h,s,v, np.array(props)])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", required=True)
    ap.add_argument("--val_split", required=True)
    args = ap.parse_args()
    Xtr_p, ytr = parse_split(args.train_split)
    Xva_p, yva = parse_split(args.val_split)
    Xtr = np.vstack([feats(p) for p in Xtr_p])
    Xva = np.vstack([feats(p) for p in Xva_p])
    clf = Pipeline([("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", C=10, gamma="scale"))])
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xva)
    print(classification_report(yva, yp, digits=4))
    print("Confusion matrix:\n", confusion_matrix(yva, yp))
if __name__ == "__main__":
    main()

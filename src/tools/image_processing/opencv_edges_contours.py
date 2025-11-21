# src/tools/opencv_edges_contours.py
import argparse, cv2, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out_prefix", default="cv_demo")
    args = ap.parse_args()

    img = cv2.imread(args.img)       # BGR
    if img is None:
        raise FileNotFoundError(args.img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edges
    edges = cv2.Canny(gray, 80, 160)
    cv2.imwrite(args.out_prefix+"_canny.png", edges)

    # Contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = img.copy()
    cv2.drawContours(cont, cnts, -1, (0,0,255), 1)
    cv2.imwrite(args.out_prefix+"_contours.png", cont)

    # Hough lines (probabilistic)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=10)
    line_img = img.copy()
    if lines is not None:
        for (x1,y1,x2,y2) in lines[:,0]:
            cv2.line(line_img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(args.out_prefix+"_lines.png", line_img)

    print("Saved:", args.out_prefix+"_canny.png", args.out_prefix+"_contours.png", args.out_prefix+"_lines.png")

if __name__ == "__main__":
    main()

import argparse, os, random, glob, pathlib
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)  # ví dụ data/rice_cls
    ap.add_argument("--out_dir", required=True)   # ví dụ data/splits
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--abs", action="store_true", help="ghi absolute path")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    classes = sorted([d for d in os.listdir(args.data_dir)
                      if os.path.isdir(os.path.join(args.data_dir, d))])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes: f.write(c+"\n")

    def all_imgs(cls):
        p = os.path.join(args.data_dir, cls)
        exts = ("*.jpg","*.jpeg","*.png","*.bmp")
        files = []
        for e in exts: files += glob.glob(os.path.join(p, e))
        return files

    tr_lines, va_lines, te_lines = [], [], []
    root = pathlib.Path(".").resolve()

    for lab, cls in enumerate(classes):
        files = all_imgs(cls)
        random.shuffle(files)
        n = len(files); nv = int(n*args.val_ratio); nt = int(n*args.test_ratio)
        tv = files[:nv]; tt = files[nv:nv+nt]; tr = files[nv+nt:]
        def fmt(x):
            if args.abs: return f"{os.path.abspath(x)} {lab}"
            # relative to project root
            p = pathlib.Path(x).resolve()
            return f"{p.relative_to(root).as_posix()} {lab}"
        tr_lines += [fmt(x) for x in tr]
        va_lines += [fmt(x) for x in tv]
        te_lines += [fmt(x) for x in tt]

    with open(os.path.join(args.out_dir,"train_cls.txt"),"w",encoding="utf-8") as f: f.write("\n".join(tr_lines))
    with open(os.path.join(args.out_dir,"val_cls.txt"),"w",encoding="utf-8") as f: f.write("\n".join(va_lines))
    with open(os.path.join(args.out_dir,"test_cls.txt"),"w",encoding="utf-8") as f: f.write("\n".join(te_lines))
    print("Done. classes:", classes)
if __name__ == "__main__":
    main()

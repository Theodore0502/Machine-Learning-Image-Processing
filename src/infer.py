import argparse, json

def infer_cls(weights, image):
    # TODO: load weights and predict class + Grad-CAM overlay
    return {"class": "demo", "score": 0.99}

def infer_seg(weights, image):
    # TODO: load weights and produce mask + %area
    return {"percent_area": 12.3}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cls","seg"], required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    args = p.parse_args()
    out = infer_cls(args.weights, args.image) if args.task=="cls" else infer_seg(args.weights, args.image)
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()

import albumentations as A

def build_cls_augment(img_size=384, use_color=True):
    aug = [A.LongestMaxSize(max_size=img_size), A.PadIfNeeded(img_size, img_size)]
    if use_color:
        aug.append(A.ColorJitter(p=0.5))
    aug.append(A.HorizontalFlip(p=0.5))
    return A.Compose(aug)

def build_seg_augment(img_size=512):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size),
        A.HorizontalFlip(p=0.5)
    ])

from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def confusion(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred, labels=labels)

def dice_coef(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

def miou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append((inter + 1e-6) / (union + 1e-6))
    return float(np.mean(ious))

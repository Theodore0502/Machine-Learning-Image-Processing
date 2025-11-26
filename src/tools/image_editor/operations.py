# src/tools/image_editor/operations.py
from PIL import Image
import numpy as np
import cv2


def adjust_brightness_contrast(
    img: Image.Image,
    brightness: int = 50,
    contrast: float = 1.0,
):
    """
    brightness: 0..100
        - 0   -> đen hoàn toàn
        - 50  -> giữ sáng gốc
        - 100 -> sáng nhất (có thể cháy sáng ở vùng sáng)

    contrast: ~ [0.5, 1.5]
        - <1  -> giảm tương phản
        - >1  -> tăng tương phản
    """
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    # 0 -> 0.0, 50 -> 1.0, 100 -> 2.0
    b_factor = brightness / 50.0
    img_cv = img_cv * b_factor

    # điều chỉnh tương phản quanh mean
    mean = img_cv.mean(axis=(0, 1), keepdims=True)
    img_cv = (img_cv - mean) * contrast + mean

    img_cv = np.clip(img_cv, 0.0, 1.0) * 255.0
    img_cv = img_cv.astype(np.uint8)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def adjust_hsv(
    img: Image.Image,
    hue_shift: int = 0,
    sat_scale: float = 1.0,
    val_scale: float = 1.0,
):
    """
    hue_shift: [-180, 180] (độ), shift vòng trên không gian H (0..179 của OpenCV)
    sat_scale: 0.0..3.0  (0 = mất màu, >1 = rất rực)
    val_scale: 0.0..3.0  (0 = đen, >1 = sáng hơn)
    """
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(img_cv)

    h = (h + hue_shift) % 180
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(out)


def flip_image(img: Image.Image, horizontal: bool = False, vertical: bool = False):
    out = img
    if horizontal:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if vertical:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out


def rotate_image(img: Image.Image, angle: int = 0):
    """angle: 0, 90, 180, 270…"""
    if angle % 360 == 0:
        return img
    return img.rotate(angle, expand=True)


def apply_all(
    img: Image.Image,
    brightness: int = 50,
    contrast: float = 1.0,
    hue_shift: int = 0,
    sat_scale: float = 1.0,
    val_scale: float = 1.0,
    angle: int = 0,
    flip_h: bool = False,
    flip_v: bool = False,
):
    # 1) sáng / tương phản
    img = adjust_brightness_contrast(img, brightness, contrast)
    # 2) hue / saturation / value
    img = adjust_hsv(img, hue_shift, sat_scale, val_scale)
    # 3) xoay / lật
    img = rotate_image(img, angle)
    img = flip_image(img, horizontal=flip_h, vertical=flip_v)
    return img

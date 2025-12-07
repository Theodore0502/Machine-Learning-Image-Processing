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
        - 0   -> tối nhất (-127.5)
        - 50  -> giữ sáng gốc
        - 100 -> sáng nhất (+127.5)

    contrast: ~ [0.5, 1.5]
        - <1  -> giảm tương phản
        - =1  -> giữ nguyên
        - >1  -> tăng tương phản
    """
    
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).astype(np.float32)

    # Step 1: Apply BRIGHTNESS FIRST
    # Brightness chuẩn: cộng offset, không nhân
    # 0 -> -127.5, 50 -> 0, 100 -> +127.5
    brightness_offset = (brightness - 50) * 2.55
    img_cv = img_cv + brightness_offset
    
    # Step 2: Apply CONTRAST AFTER (on brightness-adjusted image)
    # Điều chỉnh quanh giá trị 127.5 (giữa của [0, 255])
    img_cv = (img_cv - 127.5) * contrast + 127.5

    img_cv = np.clip(img_cv, 0.0, 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def adjust_hsv(
    img: Image.Image,
    hue_shift: int = 0,
    sat_scale: float = 1.0,
    val_scale: float = 1.0,
):
    # RGB → HSV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)

    h, s, v = cv2.split(img_cv)

    # Hue shift (OpenCV H range: 0–179, mỗi đơn vị = 2 độ)
    hue_cv_shift = int(hue_shift / 2)  # không dùng // để tránh làm tròn sai
    h = (h + hue_cv_shift) % 180

    # Scale saturation & value
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)

    # Merge & convert back
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
    img = adjust_brightness_contrast(img, brightness, contrast)

    if not (hue_shift == 0 and sat_scale == 1.0 and val_scale == 1.0):
        img = adjust_hsv(img, hue_shift, sat_scale, val_scale)

    img = rotate_image(img, angle)
    img = flip_image(img, horizontal=flip_h, vertical=flip_v)

    return img


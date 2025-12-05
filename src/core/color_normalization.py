"""
Auto Color Normalization for Rice Leaf Images

This module provides automatic color correction to normalize rice leaf images
to a standard green hue, compensating for different lighting conditions and
camera settings.

The normalization helps improve model prediction accuracy by standardizing
the input color distribution.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


def detect_dominant_hue(image: np.ndarray) -> float:
    """
    Detect the dominant hue in an image.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        
    Returns:
        Dominant hue value in degrees (0-360)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Get hue channel (0-179 in OpenCV, we'll convert to 0-360)
    hue = hsv[:, :, 0]
    
    # Calculate histogram of hue values
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    
    # Find the most common hue
    dominant_hue_cv = np.argmax(hist)
    
    # Convert from OpenCV range (0-179) to degrees (0-360)
    dominant_hue_degrees = dominant_hue_cv * 2
    
    return dominant_hue_degrees


def normalize_to_green(
    image: Image.Image, 
    target_hue: int = 120,  # Green hue
    auto_saturation: bool = True,
    auto_value: bool = True
) -> Image.Image:
    """
    Normalize image colors towards green hue (standard for healthy leaves).
    
    Args:
        image: PIL Image in RGB
        target_hue: Target hue in degrees (default 120 = green)
        auto_saturation: Whether to auto-adjust saturation
        auto_value: Whether to auto-adjust value/brightness
        
    Returns:
        Normalized PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Detect current dominant hue
    current_hue = detect_dominant_hue(img_array)
    
    # Calculate hue shift needed (in OpenCV scale: 0-179)
    target_hue_cv = target_hue / 2  # Convert degrees to OpenCV scale
    current_hue_cv = current_hue / 2
    hue_shift = target_hue_cv - current_hue_cv
    
    # Apply hue shift
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    
    # Auto-adjust saturation if needed
    if auto_saturation:
        # Calculate mean saturation
        mean_sat = np.mean(hsv[:, :, 1])
        
        # Target saturation range: 80-120 (moderate saturation)
        if mean_sat < 70:
            # Boost saturation for washed-out images
            sat_boost = 1.3
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
        elif mean_sat > 150:
            # Reduce saturation for oversaturated images
            sat_reduce = 0.85
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_reduce, 0, 255)
    
    # Auto-adjust value/brightness if needed
    if auto_value:
        # Calculate mean brightness
        mean_val = np.mean(hsv[:, :, 2])
        
        # Target brightness range: 100-180 (well-lit)
        if mean_val < 90:
            # Brighten dark images
            val_boost = 1.25
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_boost, 0, 255)
        elif mean_val > 200:
            # Darken overexposed images
            val_reduce = 0.9
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_reduce, 0, 255)
    
    # Convert back to uint8 and RGB
    hsv = hsv.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return Image.fromarray(rgb)


def auto_normalize_leaf(
    image: Image.Image,
    apply_hue_correction: bool = True,
    apply_saturation_correction: bool = True,
    apply_brightness_correction: bool = True
) -> Tuple[Image.Image, dict]:
    """
    Automatically normalize a rice leaf image with detailed statistics.
    
    Args:
        image: PIL Image to normalize
        apply_hue_correction: Whether to correct hue towards green
        apply_saturation_correction: Whether to auto-adjust saturation
        apply_brightness_correction: Whether to auto-adjust brightness
        
    Returns:
        Tuple of (normalized_image, stats_dict)
    """
    # Get original statistics
    img_array = np.array(image)
    hsv_original = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    original_stats = {
        'dominant_hue': detect_dominant_hue(img_array),
        'mean_saturation': float(np.mean(hsv_original[:, :, 1])),
        'mean_brightness': float(np.mean(hsv_original[:, :, 2]))
    }
    
    # Normalize if corrections are enabled
    if apply_hue_correction or apply_saturation_correction or apply_brightness_correction:
        normalized = normalize_to_green(
            image,
            auto_saturation=apply_saturation_correction,
            auto_value=apply_brightness_correction
        )
    else:
        normalized = image
    
    # Get normalized statistics
    norm_array = np.array(normalized)
    hsv_normalized = cv2.cvtColor(norm_array, cv2.COLOR_RGB2HSV)
    
    normalized_stats = {
        'dominant_hue': detect_dominant_hue(norm_array),
        'mean_saturation': float(np.mean(hsv_normalized[:, :, 1])),
        'mean_brightness': float(np.mean(hsv_normalized[:, :, 2]))
    }
    
    stats = {
        'original': original_stats,
        'normalized': normalized_stats,
        'corrections_applied': {
            'hue': apply_hue_correction,
            'saturation': apply_saturation_correction,
            'brightness': apply_brightness_correction
        }
    }
    
    return normalized, stats


def get_normalization_message(stats: dict) -> str:
    """
    Generate a user-friendly message explaining the normalization applied.
    
    Args:
        stats: Statistics dictionary from auto_normalize_leaf
        
    Returns:
        Markdown formatted message
    """
    orig = stats['original']
    norm = stats['normalized']
    
    hue_change = norm['dominant_hue'] - orig['dominant_hue']
    sat_change = norm['mean_saturation'] - orig['mean_saturation']
    bright_change = norm['mean_brightness'] - orig['mean_brightness']
    
    message = "### 🎨 Auto Color Normalization Applied\n\n"
    
    if abs(hue_change) > 5:
        message += f"- **Hue**: {orig['dominant_hue']:.0f}° → {norm['dominant_hue']:.0f}° "
        message += f"({'shifted towards green ✅' if hue_change > 0 else 'adjusted'})\n"
    
    if abs(sat_change) > 5:
        change_desc = "increased" if sat_change > 0 else "decreased"
        message += f"- **Saturation**: {change_desc} by {abs(sat_change):.0f}\n"
    
    if abs(bright_change) > 5:
        change_desc = "brightened" if bright_change > 0 else "darkened"
        message += f"- **Brightness**: {change_desc} by {abs(bright_change):.0f}\n"
    
    if abs(hue_change) < 5 and abs(sat_change) < 5 and abs(bright_change) < 5:
        message += "- Image already well-normalized ✅\n"
    
    return message

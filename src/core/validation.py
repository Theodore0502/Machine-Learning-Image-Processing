"""
Rice Leaf Validation Module

Validates whether predictions are likely valid rice leaf images
based on model confidence thresholds.
"""

from typing import Tuple
import numpy as np


# Confidence thresholds
CONFIDENCE_THRESHOLD_LOW = 0.60  # Below this = uncertain/not rice leaf
CONFIDENCE_THRESHOLD_HIGH = 0.80  # Above this = high confidence


def is_rice_leaf(probabilities: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD_LOW) -> Tuple[bool, str, str]:
    """
    Determine if the image is likely a rice leaf based on prediction confidence.
    
    Args:
        probabilities: Array of class probabilities [prob1, prob2, ..., probN]
        threshold: Minimum confidence to consider as valid rice leaf
        
    Returns:
        Tuple of:
            - is_valid (bool): Whether image is likely a rice leaf
            - confidence_level (str): "high", "medium", or "low"
            - message (str): User-friendly warning/info message
    """
    max_confidence = float(np.max(probabilities))
    
    # Determine confidence level
    if max_confidence >= CONFIDENCE_THRESHOLD_HIGH:
        confidence_level = "high"
        badge = "🟢"
        message = f"{badge} **High Confidence** ({max_confidence*100:.1f}%)"
        is_valid = True
        
    elif max_confidence >= CONFIDENCE_THRESHOLD_LOW:
        confidence_level = "medium"
        badge = "🟡"
        message = f"{badge} **Medium Confidence** ({max_confidence*100:.1f}%)"
        is_valid = True
        
    else:
        confidence_level = "low"
        badge = "🔴"
        message = (
            f"{badge} **Low Confidence** ({max_confidence*100:.1f}%)\n\n"
            f"⚠️ **Warning:** This may not be a rice leaf or the image quality is poor. "
            f"Please upload a clearer rice leaf image for better results."
        )
        is_valid = False
    
    return is_valid, confidence_level, message


def get_confidence_badge(confidence: float) -> str:
    """
    Get a color-coded emoji badge based on confidence level.
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
        
    Returns:
        Emoji badge string
    """
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return "🟢"
    elif confidence >= CONFIDENCE_THRESHOLD_LOW:
        return "🟡"
    else:
        return "🔴"


def format_confidence_message(confidence: float, class_name: str) -> str:
    """
    Format a confidence message with appropriate styling based on level.
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
        class_name: Predicted class name
        
    Returns:
        Formatted markdown string
    """
    badge = get_confidence_badge(confidence)
    
    if confidence >= CONFIDENCE_THRESHOLD_HIGH:
        return f"{badge} **High confidence** prediction: `{class_name}` ({confidence*100:.2f}%)"
    elif confidence >= CONFIDENCE_THRESHOLD_LOW:
        return f"{badge} **Medium confidence** prediction: `{class_name}` ({confidence*100:.2f}%)"
    else:
        return (
            f"{badge} **Low confidence** prediction: `{class_name}` ({confidence*100:.2f}%)\n\n"
            f"⚠️ This might not be a rice leaf image."
        )

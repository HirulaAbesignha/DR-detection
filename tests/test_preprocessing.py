"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import normalize_image, preprocess_image


def test_normalize_image():
    """Test image normalization."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    normalized = normalize_image(img)
    
    assert normalized.shape == img.shape
    assert normalized.dtype in [np.float32, np.float64]


def test_preprocess_image():
    """Test complete preprocessing pipeline."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    processed = preprocess_image(img, target_size=(224, 224))
    
    assert processed.shape == (224, 224, 3)
    assert processed.dtype in [np.float32, np.float64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
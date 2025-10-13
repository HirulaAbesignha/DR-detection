"""
Unit tests for model module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import create_dr_model, compile_model


def test_create_model():
    """Test model creation."""
    model, base_model = create_dr_model()
    
    assert model is not None
    assert base_model is not None
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 5)


def test_model_prediction():
    """Test model can make predictions."""
    model, _ = create_dr_model()
    model = compile_model(model)
    
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    
    assert predictions.shape == (1, 5)
    assert np.allclose(predictions.sum(), 1.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
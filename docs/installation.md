# Complete Setup Instructions

## Step-by-Step Setup Guide for Local Machine

### 1. Create Project Structure

Create the following folder structure:

```
diabetic-retinopathy-detection/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── data_loader.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── gradcam.py
│   ├── report.py
│   └── app.py
├── data/
│   └── retinal_images/
│       ├── class_0/
│       ├── class_1/
│       ├── class_2/
│       ├── class_3/
│       └── class_4/
├── models/
├── outputs/
│   ├── plots/
│   ├── logs/
│   └── predictions/
├── configs/
├── tests/
├── notebooks/
├── docs/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### 2. Install Python and Dependencies

**Windows:**
```bash
# Install Python 3.10 from python.org

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install GPU Support (Optional but Recommended)

**For NVIDIA GPU:**

1. Install CUDA Toolkit 11.8:
   - Download from: https://developer.nvidia.com/cuda-downloads

2. Install cuDNN 8.6:
   - Download from: https://developer.nvidia.com/cudnn

3. Verify installation:
```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 4. Prepare Dataset

**Option A: Use Local Dataset**

Place images in the data/retinal_images/ folder following this structure:
```
data/retinal_images/
├── class_0/  # No DR images
├── class_1/  # Mild DR images
├── class_2/  # Moderate DR images
├── class_3/  # Severe DR images
└── class_4/  # Proliferative DR images
```

**Option B: Download from Kaggle**

```bash
pip install kaggle

# Set up Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

kaggle datasets download -d your-dataset-name
unzip your-dataset-name.zip -d data/retinal_images/
```

**Option C: Auto-download from HuggingFace (Default)**

The system will automatically download the dataset on first run.

### 5. Training the Model

**Basic Training:**
```bash
python -m src.train
```

**Advanced Training:**
```bash
python -m src.train \
    --data-path ./data/retinal_images \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --sample-size 5000
```

**Expected Output:**
- Model saved to: models/best_dr_model.h5
- Training plots in: outputs/plots/
- Training logs in: outputs/logs/

### 6. Making Predictions

**Single Image:**
```bash
python -m src.predict \
    --image path/to/image.jpg \
    --model models/best_dr_model.h5 \
    --report \
    --save-visualization result.png
```

**Batch Prediction:**
```bash
python -m src.predict \
    --folder path/to/images/ \
    --model models/best_dr_model.h5 \
    --output predictions.csv
```

### 7. Launch Web Interface

```bash
python -m src.app --model models/best_dr_model.h5 --share
```

Access at: http://localhost:7860

### 8. Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Configuration

### Adjusting for Your Hardware

Edit `src/utils.py` CONFIG dictionary:

**For 8GB GPU:**
```python
CONFIG = {
    'BATCH_SIZE': 8,
    'SAMPLE_SIZE': 5000,
    ...
}
```

**For 6GB GPU:**
```python
CONFIG = {
    'BATCH_SIZE': 4,
    'SAMPLE_SIZE': 3000,
    ...
}
```

**For CPU Only:**
```python
CONFIG = {
    'BATCH_SIZE': 2,
    'SAMPLE_SIZE': 1000,
    ...
}
```

### Memory Optimization

In `src/utils.py`, adjust GPU memory limit:
```python
def setup_gpu(memory_limit_mb=7168):  # Adjust this value
    ...
```

## Troubleshooting

### Issue: GPU Out of Memory

**Solution:**
1. Reduce batch size in CONFIG
2. Reduce image size
3. Reduce sample size

### Issue: Module Not Found

**Solution:**
```bash
# Make sure you're in project root
cd diabetic-retinopathy-detection

# Install in development mode
pip install -e .
```

### Issue: CUDA Not Found

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: Dataset Not Loading

**Solution:**
1. Check internet connection (for HuggingFace)
2. Use local dataset instead
3. Verify folder structure

### Issue: Gradio Interface Not Working

**Solution:**
```bash
# Reinstall Gradio
pip uninstall gradio
pip install gradio

# Try different port
python -m src.app --port 7861
```

## File Descriptions

### Core Files

- `src/__init__.py` - Package initialization
- `src/utils.py` - Utility functions and configuration
- `src/model.py` - Model architecture definitions
- `src/preprocessing.py` - Image preprocessing pipeline
- `src/augmentation.py` - Data augmentation functions
- `src/data_loader.py` - Data loading and generators
- `src/train.py` - Training pipeline
- `src/predict.py` - Prediction utilities
- `src/evaluation.py` - Model evaluation metrics
- `src/visualization.py` - Plotting functions
- `src/gradcam.py` - Grad-CAM implementation
- `src/report.py` - Medical report generation
- `src/app.py` - Gradio web interface

### Configuration Files

- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `README.md` - Project documentation

## Quick Start Commands

```bash
# 1. Setup
git clone <your-repo>
cd diabetic-retinopathy-detection
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Train
python -m src.train

# 3. Predict
python -m src.predict --image test.jpg --model models/best_dr_model.h5 --report

# 4. Launch Web Interface
python -m src.app --model models/best_dr_model.h5
```

## Next Steps

1. Train model with your own dataset
2. Evaluate model performance
3. Fine-tune hyperparameters
4. Deploy web interface
5. Integrate with your application

## Support

For issues and questions:
- Check troubleshooting section above
- Review README.md for detailed documentation
- Open an issue on GitHub

## Additional Resources

- TensorFlow documentation: https://tensorflow.org
- Keras documentation: https://keras.io
- Gradio documentation: https://gradio.app
- Project repository: [Your GitHub URL]
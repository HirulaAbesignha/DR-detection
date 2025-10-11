# Diabetic Retinopathy Detection System

Advanced deep learning system for automated detection and classification of diabetic retinopathy severity from retinal fundus images using transfer learning and computer vision techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [API Reference](#api-reference)
- [Model Export](#model-export)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Disclaimer](#disclaimer)

## Overview

Diabetic retinopathy is a diabetes complication that affects eyes and is a leading cause of blindness among working-age adults. Early detection and treatment can prevent vision loss. This project implements a state-of-the-art deep learning solution for automated screening and classification of diabetic retinopathy from retinal fundus photographs.

The system classifies images into five severity levels based on the International Clinical Diabetic Retinopathy Disease Severity Scale:

- **Class 0: No DR** - No abnormalities detected
- **Class 1: Mild DR** - Microaneurysms only
- **Class 2: Moderate DR** - More than just microaneurysms but less than Severe DR
- **Class 3: Severe DR** - More than 20 intraretinal hemorrhages in each of 4 quadrants; definite venous beading in 2+ quadrants; prominent IRMA in 1+ quadrant
- **Class 4: Proliferative DR** - Neovascularization, vitreous or preretinal hemorrhage

### Key Capabilities

- Automated multi-class classification of diabetic retinopathy severity
- High-accuracy predictions with confidence scores
- Interpretable results using Grad-CAM visualization
- Robust preprocessing pipeline optimized for retinal images
- Production-ready inference system
- Web-based interface for easy deployment
- Comprehensive evaluation metrics and reports

## Features

### Core Functionality

- End-to-end deep learning pipeline from data loading to model deployment
- Transfer learning utilizing EfficientNetB3 pre-trained on ImageNet
- Advanced image preprocessing with CLAHE enhancement and adaptive normalization
- Sophisticated data augmentation strategies for handling class imbalance
- Grad-CAM heatmaps showing regions of interest for predictions
- Multi-format support for JPEG, PNG, and various image formats
- Efficient batch processing for multiple images

### Technical Features

- Memory-optimized data generators for large datasets
- GPU acceleration with automatic memory management
- Model checkpointing, early stopping, and learning rate scheduling
- Comprehensive logging of training metrics and validation scores
- Support for multiple model versions and comparison
- Mixed precision training support for faster computation

### User Interface

- Gradio web interface for interactive predictions
- Medical report generation with clinical recommendations
- Training visualization dashboard with curves and matrices
- Full command-line interface support

### Production Features

- Model export in TensorFlow SavedModel, HDF5, and TensorFlow Lite formats
- RESTful API ready architecture
- Docker containerization support
- Scalable deployment design
- Robust error handling and logging

## System Requirements

### Hardware Requirements

#### Minimum Configuration
- CPU: Intel Core i5 or AMD equivalent (4 cores)
- RAM: 16 GB
- GPU: NVIDIA GPU with 6 GB VRAM
- Storage: 20 GB free space

#### Recommended Configuration
- CPU: Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores)
- RAM: 32 GB or more
- GPU: NVIDIA RTX 3070/4070 or better (8+ GB VRAM)
- Storage: 50 GB SSD storage

### Software Requirements

#### Operating System
- Linux: Ubuntu 20.04 LTS or later (recommended)
- Windows: Windows 10/11 (64-bit)
- macOS: macOS 11.0 or later (CPU only)

#### Core Dependencies
- Python: 3.8, 3.9, 3.10, or 3.11
- CUDA: 11.2 or higher (for GPU support)
- cuDNN: 8.1 or higher (for GPU support)
- TensorFlow: 2.12.0 or higher

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/HirulaAbesignha/DR-detection.git
cd diabetic-retinopathy-detection
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n dr_detection python=3.10
conda activate dr_detection
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## Dataset Preparation

### Option 1: Automatic Download from HuggingFace

The system automatically downloads the dataset on first run. Dataset will be streamed from HuggingFace Datasets Hub.

### Option 2: Use Local Dataset

Prepare your dataset in the following directory structure:

```
data/retinal_images/
├── class_0/          # No DR images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_1/          # Mild DR images
├── class_2/          # Moderate DR images
├── class_3/          # Severe DR images
└── class_4/          # Proliferative DR images
```

**Image Requirements:**
- Format: JPEG, PNG, or TIFF
- Resolution: Minimum 224x224 pixels
- Color Space: RGB color images
- Quality: Good quality retinal fundus photographs

## Usage

### Training a Model

#### Basic Training

```bash
python src/train.py
```

#### Advanced Training with Options

```bash
python src/train.py \
    --data-path ./data/retinal_images \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.0001 \
    --output-dir ./outputs
```

**Available Arguments:**

- `--data-path`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Initial learning rate (default: 1e-4)
- `--img-size`: Input image size (default: 224)
- `--output-dir`: Directory for outputs

### Making Predictions

#### Single Image Prediction

```bash
python src/predict.py \
    --image path/to/retinal_image.jpg \
    --model models/best_dr_model.h5 \
    --output prediction_result.json
```

#### Batch Prediction

```bash
python src/predict.py \
    --folder path/to/images/ \
    --model models/best_dr_model.h5 \
    --output predictions.csv
```

### Web Interface

Launch the Gradio interface:

```bash
python src/app.py --model models/best_dr_model.h5
```

Access the interface at: `http://localhost:7860`

The interface provides:
- Image upload functionality
- Real-time prediction with confidence scores
- Grad-CAM visualization showing attention regions
- Detailed medical report with recommendations
- Probability distribution for all classes

## Model Architecture

The model uses EfficientNetB3 as a feature extractor with custom classification head:

**Base Model:**
- EfficientNetB3 pre-trained on ImageNet
- Input shape: 224x224x3

**Custom Classification Head:**
- Dense Layer 1: 512 units, ReLU activation, Batch Normalization, Dropout 0.5
- Dense Layer 2: 256 units, ReLU activation, Batch Normalization, Dropout 0.4
- Dense Layer 3: 128 units, ReLU activation, Dropout 0.3
- Output Layer: 5 units, Softmax activation

**Total Parameters:** Approximately 12 million trainable parameters

**Training Strategy:**
1. Stage 1: Train with frozen base model (10-20 epochs)
2. Stage 2: Fine-tune top layers of base model (10-20 epochs)

## Performance Metrics

Expected performance on test set:

- Accuracy: 85-92%
- AUC-ROC: 0.92-0.96
- Precision: 0.85-0.90
- Recall: 0.83-0.89
- F1-Score: 0.84-0.89

Performance varies based on dataset quality and size. Results are computed using stratified test set with class-balanced evaluation.

## Data Preprocessing Pipeline

The preprocessing pipeline consists of multiple stages:

### Stage 1: Data Loading and Validation
- Load images from local storage or streaming source
- Validate image format and integrity
- Check minimum resolution requirements
- Identify and handle missing or corrupted files

### Stage 2: Missing Data Analysis
- Analyze dataset completeness
- Generate missing data statistics
- Create visualization reports
- Handle missing labels or images

### Stage 3: Outlier Detection
- Detect outliers using Interquartile Range (IQR) method
- Analyze brightness, contrast, and image dimensions
- Flag anomalous images for review
- Generate outlier distribution plots

### Stage 4: Class Imbalance Handling
- Compute class distribution statistics
- Calculate imbalance ratios
- Apply class-weighted loss functions
- Implement targeted data augmentation

### Stage 5: Image Enhancement
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Enhance retinal vessel visibility
- Normalize color channels
- Preserve important medical features

### Stage 6: Normalization
- Resize images to target dimensions (224x224)
- Scale pixel values to [0, 1] range
- Apply ImageNet normalization statistics
- Ensure consistent color space (RGB)

### Stage 7: Data Augmentation
- Horizontal flipping (50% probability)
- Random rotation (±15 degrees, 30% probability)
- Brightness adjustment (±20%, 30% probability)
- Class-specific augmentation rates
- Preserve medical relevance of augmentations

## Training Pipeline

### Data Splitting

Stratified splitting to maintain class distribution:
- Training Set: 70% of total data
- Validation Set: 15% of total data
- Test Set: 15% of total data

### Training Configuration

**Optimizer:** Adam with learning rate 1e-4

**Loss Function:** Categorical cross-entropy with class weights

**Callbacks:**
- ModelCheckpoint: Save best model based on validation accuracy
- EarlyStopping: Stop training if validation loss plateaus (patience: 10 epochs)
- ReduceLROnPlateau: Reduce learning rate by 0.5 when validation loss plateaus (patience: 5 epochs)
- CSVLogger: Log all metrics to CSV file
- TerminateOnNaN: Stop training if NaN loss detected

**Training Stages:**

1. **Frozen Base Training (Stage 1):**
   - Freeze EfficientNetB3 base model
   - Train only custom classification head
   - Duration: 15-20 epochs
   - Learning rate: 1e-4

2. **Fine-tuning (Stage 2):**
   - Unfreeze top layers of base model
   - Continue training with lower learning rate
   - Duration: 10-15 epochs
   - Learning rate: 1e-5

### Memory Optimization

- Batch-wise data loading using custom generators
- Automatic garbage collection between epochs
- GPU memory growth enabled
- Efficient tensor operations

## Evaluation and Visualization

### Metrics Computed

**Classification Metrics:**
- Per-class accuracy, precision, recall, F1-score
- Overall accuracy and balanced accuracy
- Cohen's Kappa score
- Matthews Correlation Coefficient

**Probabilistic Metrics:**
- Multi-class AUC-ROC
- Average Precision Score
- Log Loss (Cross-Entropy)

**Confusion Matrix:**
- True positives, false positives, true negatives, false negatives for each class
- Normalized confusion matrix showing percentages

### Visualizations Generated

1. **Training History Plots:**
   - Accuracy curves (train vs validation)
   - Loss curves (train vs validation)
   - AUC curves (train vs validation)
   - Learning rate schedule

2. **Confusion Matrix:**
   - Heatmap visualization
   - Raw counts and percentages
   - Per-class performance analysis

3. **ROC Curves:**
   - One-vs-Rest ROC curves for each class
   - Area Under Curve (AUC) scores
   - Comparison across all classes

4. **Precision-Recall Curves:**
   - Per-class precision-recall curves
   - Average Precision (AP) scores

5. **Grad-CAM Visualizations:**
   - Attention heatmaps overlaid on original images
   - Class-specific activation regions
   - Model interpretability analysis

All visualizations are saved in the outputs directory with high resolution (300 DPI) for publication quality.

## API Reference

### Training API

```python
from src.train import train_model

model, history, metrics = train_model(
    data_path='./data/retinal_images',
    epochs=50,
    batch_size=8,
    learning_rate=1e-4
)
```

**Parameters:**
- `data_path` (str): Path to dataset directory
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `learning_rate` (float): Initial learning rate

**Returns:**
- `model`: Trained Keras model
- `history`: Training history object
- `metrics`: Dictionary containing test metrics

### Prediction API

```python
from src.predict import predict_single_image

result = predict_single_image(
    image='path/to/image.jpg',
    model=model,
    return_gradcam=True
)
```

**Parameters:**
- `image` (str or numpy.ndarray): Path to image or image array
- `model`: Trained Keras model
- `return_gradcam` (bool): Whether to generate Grad-CAM visualization

**Returns:**
- `result` (dict): Dictionary containing:
  - `class` (int): Predicted class index
  - `class_name` (str): Predicted class name
  - `confidence` (float): Prediction confidence
  - `all_probabilities` (dict): Probabilities for all classes
  - `gradcam` (numpy.ndarray): Grad-CAM heatmap (if requested)
  - `original` (numpy.ndarray): Original processed image

### Batch Prediction API

```python
from src.predict import batch_predict

results_df = batch_predict(
    image_folder='path/to/folder',
    model=model,
    output_csv='predictions.csv'
)
```

**Parameters:**
- `image_folder` (str): Path to folder containing images
- `model`: Trained Keras model
- `output_csv` (str): Path to save CSV results

**Returns:**
- `results_df` (pandas.DataFrame): DataFrame with predictions

### Model Loading API

```python
from src.utils import load_trained_model

model = load_trained_model('models/best_dr_model.h5')
```

**Parameters:**
- `model_path` (str): Path to saved model file

**Returns:**
- `model`: Loaded Keras model

### Medical Report Generation API

```python
from src.report import generate_medical_report

report = generate_medical_report(prediction_result)
print(report)
```

**Parameters:**
- `prediction_result` (dict): Prediction result from predict_single_image

**Returns:**
- `report` (str): Formatted medical report with recommendations

## Model Export

Export trained models for deployment:

```bash
python src/export_model.py --model models/best_dr_model.h5 --output exports/
```

**Exported Formats:**

1. **TensorFlow SavedModel:**
   - Full model with graph and weights
   - Compatible with TensorFlow Serving
   - Location: `exports/saved_model/`

2. **HDF5 Format:**
   - Keras native format
   - Easy to load and use
   - Location: `exports/model.h5`

3. **TensorFlow Lite:**
   - Optimized for mobile and edge devices
   - Reduced model size
   - Location: `exports/model.tflite`

4. **Configuration File:**
   - Model metadata and settings
   - Location: `exports/config.json`

## Project Structure

```
diabetic-retinopathy-detection/
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── LICENSE                         # Project license
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── train.py                   # Training script
│   ├── predict.py                 # Prediction script
│   ├── app.py                     # Gradio web interface
│   ├── model.py                   # Model architecture
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Preprocessing pipeline
│   ├── augmentation.py            # Data augmentation
│   ├── evaluation.py              # Evaluation metrics
│   ├── visualization.py           # Plotting functions
│   ├── gradcam.py                 # Grad-CAM implementation
│   ├── report.py                  # Report generation
│   ├── export_model.py            # Model export utilities
│   └── utils.py                   # Helper functions
│
├── configs/                       # Configuration files
│   ├── model_config.yaml          # Model configuration
│   └── training_config.yaml       # Training configuration
│
├── data/                          # Dataset directory
│   ├── retinal_images/            # Image data
│   └── README.md                  # Dataset information
│
├── models/                        # Saved models
│   ├── best_dr_model.h5           # Best trained model
│   ├── checkpoint_*.h5            # Training checkpoints
│   └── config.json                # Model configuration
│
├── outputs/                       # Training outputs
│   ├── plots/                     # Visualization plots
│   ├── logs/                      # Training logs
│   └── predictions/               # Prediction results
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_analysis.ipynb
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_prediction.py
│
└── docs/                          # Documentation
    ├── installation.md
    ├── usage.md
    └── api_reference.md
```

## Configuration

### Model Configuration (configs/model_config.yaml)

```yaml
model:
  architecture: efficientnetb3
  input_size: 224
  num_classes: 5
  dropout_rate: 0.5
  weights: imagenet

training:
  batch_size: 8
  epochs: 50
  learning_rate: 0.0001
  optimizer: adam
  loss: categorical_crossentropy

augmentation:
  horizontal_flip: true
  rotation_range: 15
  brightness_range: [0.8, 1.2]
  zoom_range: 0.1
```

### Training Configuration (configs/training_config.yaml)

```yaml
data:
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  shuffle: true
  seed: 42

callbacks:
  early_stopping:
    monitor: val_loss
    patience: 10
    restore_best_weights: true
  
  reduce_lr:
    monitor: val_loss
    factor: 0.5
    patience: 5
    min_lr: 0.0000001
  
  model_checkpoint:
    monitor: val_accuracy
    save_best_only: true
    mode: max

output:
  model_dir: ./models
  log_dir: ./outputs/logs
  plot_dir: ./outputs/plots
```

## Testing

Run unit tests to verify functionality:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage report
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

**Test Coverage:**
- Data loading and preprocessing
- Model architecture and compilation
- Prediction pipeline
- Grad-CAM visualization
- Report generation
- Export functionality

## Deployment

### Local Deployment

```bash
# Start the web interface
python src/app.py --model models/best_dr_model.h5 --port 7860
```

### Docker Deployment

```bash
# Build Docker image
docker build -t dr-detection:latest .

# Run container
docker run -p 7860:7860 dr-detection:latest
```

### Cloud Deployment

**AWS SageMaker:**
```bash
# Package model
tar -czf model.tar.gz models/

# Upload to S3 and deploy
aws sagemaker create-model --model-name dr-detection-model
```

**Google Cloud AI Platform:**
```bash
# Deploy model
gcloud ai-platform models create dr_detection
gcloud ai-platform versions create v1 --model dr_detection --origin=gs://bucket/model/
```

### API Deployment

Create a Flask/FastAPI server:

```python
from flask import Flask, request, jsonify
from src.predict import predict_single_image
from src.utils import load_trained_model

app = Flask(__name__)
model = load_trained_model('models/best_dr_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = predict_single_image(file, model)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Troubleshooting

### Common Issues

**1. GPU Out of Memory:**
```
Solution: Reduce batch size in CONFIG
CONFIG['BATCH_SIZE'] = 4  # Reduce from 8
```

**2. CUDA Not Found:**
```
Solution: Verify CUDA installation
nvidia-smi
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**3. Model Loading Errors:**
```
Solution: Ensure custom objects are specified
model = load_model('model.h5', custom_objects={...})
```

**4. Dataset Download Failures:**
```
Solution: Use local dataset or check internet connection
Place images in data/retinal_images/ following directory structure
```

**5. Gradio Interface Not Loading:**
```
Solution: Check port availability and firewall settings
Use different port: python src/app.py --port 7861
```

### Performance Optimization

**For 8GB GPU:**
- Use batch_size = 4 or 8
- Disable mixed precision training
- Enable gradient accumulation

**For Limited RAM:**
- Use data generators instead of loading all data
- Reduce SAMPLE_SIZE in CONFIG
- Enable memory cleanup after epochs

**For Faster Training:**
- Use mixed precision training (if GPU supports)
- Increase batch size if GPU memory allows
- Use multiple workers for data loading

## Contributing

We welcome contributions to improve this project. Please follow these guidelines:

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/improvement`)
8. Create a Pull Request

### Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Comment complex logic

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Keep first line under 50 characters
- Provide detailed description if necessary

## Known Issues

1. **Grad-CAM Compatibility:** Grad-CAM may fail on some EfficientNet variants. Fallback to original image if error occurs.

2. **Large Batch Sizes:** Batch sizes above 16 may cause OOM errors on 8GB GPUs. Use batch_size=8 for stability.

3. **Dataset Streaming:** Requires stable internet connection. Use local dataset for offline training.

4. **Mixed Precision:** May cause numerical instability on older GPUs. Disable if experiencing NaN losses.

5. **Windows Path Issues:** Use forward slashes (/) in paths for cross-platform compatibility.

## Future Improvements

### Planned Features

- Multi-GPU distributed training support
- Model ensemble techniques for improved accuracy
- Active learning pipeline for continuous improvement
- ONNX export for cross-platform deployment
- REST API with authentication
- Real-time video stream analysis
- Integration with PACS systems
- Automated hyperparameter tuning
- Explainable AI features beyond Grad-CAM
- Mobile application development

### Research Directions

- Attention mechanisms for better feature extraction
- Self-supervised pre-training on retinal images
- Few-shot learning for rare severity classes
- Multi-task learning (DR + other retinal diseases)
- Generative models for data augmentation
- Uncertainty quantification in predictions

## Citation

If you use this code in your research, please cite:

```bibtex
@software{diabetic_retinopathy_detection_2025,
  author = {Hirula Abeisingha},
  title = {Diabetic Retinopathy Detection System: Deep Learning Approach},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HirulaAbesignha/DR-detection},
  version = {1.0.0}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for full details.

```
MIT License

Copyright (c) 2025 Hirula Abesingha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "DR-detection"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

### Dataset
- Diabetic Retinopathy Detection Dataset from HuggingFace
- Original dataset curated by medical professionals
- Preprocessing and annotations by community contributors

### Libraries and Frameworks
- TensorFlow/Keras: Deep learning framework
- EfficientNet: Base model architecture (Google Research)
- OpenCV: Image processing operations
- Scikit-learn: Machine learning utilities
- Gradio: Web interface framework
- NumPy, Pandas: Data manipulation
- Matplotlib, Seaborn: Visualization

### Inspiration
- Kaggle Diabetic Retinopathy Detection Challenge
- Medical image analysis research community
- Open-source deep learning projects

### Contributors
- Hirula Abesignha - Initial development
- Contributors list will be maintained in CONTRIBUTORS.md

## Contact

### Project Maintainer
- Name: Hirula Abesignha
- Email: hirulapinibinda01@gmail.com
- GitHub: [@yourusername](https://github.com/HirulaAbesignha)
- LinkedIn: [Your Profile](https://linkedin.com/in/hirulaabesignha)

### Support
- Issue Tracker: [GitHub Issues](https://github.com/HirulaAbesignha/DR-detection/issues)
- Discussions: [GitHub Discussions](https://github.com/HirulaAbesignha/DR-detection/discussions)
- Documentation: [Project Wiki](https://github.com/HirulaAbesignha/DR-detection/wiki)

### Community
- Join our Discord server for discussions and support
- Follow development updates on Twitter
- Subscribe to our mailing list for announcements

## Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This software is provided for research and educational purposes only. It is NOT intended for clinical use or medical diagnosis.

### Important Points

1. **Not a Medical Device:** This system is not FDA-approved or certified as a medical device. It should not be used for clinical decision-making.

2. **No Medical Advice:** Results from this system do not constitute medical advice. Always consult with qualified healthcare professionals and ophthalmologists for diagnosis and treatment.

3. **Screening Tool Only:** This system may be used as a screening tool to identify cases requiring further examination by medical professionals, but should never replace professional medical evaluation.

4. **Accuracy Limitations:** While the system achieves high accuracy on test datasets, real-world performance may vary. False positives and false negatives can occur.

5. **No Liability:** The developers and contributors assume no liability for any medical decisions made based on this system's output.

6. **Regulatory Compliance:** Users are responsible for ensuring compliance with local regulations and medical device laws in their jurisdiction.

7. **Data Privacy:** Users must ensure compliance with HIPAA, GDPR, and other relevant data protection regulations when handling patient data.

8. **Validation Required:** Any clinical deployment requires thorough validation, regulatory approval, and oversight by medical professionals.

### Recommended Use

- Academic research and education
- Algorithm development and testing
- Non-clinical screening programs under medical supervision
- Technology demonstration and proof-of-concept
- Training for medical AI development

### Professional Medical Consultation Required

All individuals should seek professional medical evaluation for:
- Definitive diagnosis of diabetic retinopathy
- Treatment recommendations
- Disease monitoring and management
- Any vision-related symptoms or concerns

By using this software, you acknowledge that you have read, understood, and agree to this disclaimer.

---

**Version:** 1.0.0  
**Last Updated:** 2025  
**Maintained by:** Hirula Abesignha

For the latest updates and releases, visit: [GitHub Repository](https://github.com/HirulaAbesignha/DR-detection)

[Back to Top](#diabeticretinopathydetectionsystem)
"""
Prediction utilities for DR detection.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path

from .utils import CONFIG, load_trained_model
from .preprocessing import preprocess_image
from .gradcam import apply_gradcam
from .report import generate_medical_report


def predict_single_image(image, model, return_gradcam=False):
    """Predict diabetic retinopathy severity for a single image."""
    try:
        # Load and preprocess image
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            original = np.array(image)
        else:
            original = np.array(image)
        
        # Store original
        original_image = original.copy()
        
        # Preprocess
        processed_image = preprocess_image(original)
        
        # Predict
        img_array = np.expand_dims(processed_image, axis=0)
        predictions = model.predict(img_array, verbose=0)
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        result = {
            'class': predicted_class,
            'class_name': CONFIG['CLASS_NAMES'][predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                CONFIG['CLASS_NAMES'][i]: float(predictions[0][i])
                for i in range(5)
            }
        }
        
        # Generate Grad-CAM if requested
        if return_gradcam:
            try:
                gradcam_img = apply_gradcam(processed_image, model)
                result['gradcam'] = gradcam_img
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                result['gradcam'] = cv2.resize(original, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
        
        # Store processed image
        result['original'] = cv2.resize(original_image, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise


def batch_predict(image_folder, model, output_csv='predictions.csv', batch_size=16):
    """Predict on a batch of images from a folder."""
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(Path(image_folder).glob(ext)))
    
    print(f"Found {len(image_files)} images in {image_folder}")
    
    results = []
    
    for i, img_path in enumerate(image_files):
        try:
            # Predict
            result = predict_single_image(str(img_path), model, return_gradcam=False)
            
            # Store result
            result_dict = {
                'filename': img_path.name,
                'predicted_class': result['class'],
                'class_name': result['class_name'],
                'confidence': result['confidence']
            }
            
            # Add all probabilities
            for class_name, prob in result['all_probabilities'].items():
                result_dict[f'prob_{class_name.replace(" ", "_")}'] = prob
            
            results.append(result_dict)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                'filename': img_path.name,
                'predicted_class': -1,
                'class_name': 'ERROR',
                'confidence': 0.0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(df['class_name'].value_counts())
    
    return df


def predict_with_report(image_path, model, save_visualization=None):
    """Predict and generate detailed medical report."""
    # Make prediction
    result = predict_single_image(image_path, model, return_gradcam=True)
    
    # Generate medical report
    report = generate_medical_report(result)
    
    # Save visualization if requested
    if save_visualization:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(result['original'])
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM
        if 'gradcam' in result:
            axes[1].imshow(result['gradcam'])
            axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        
        # Probabilities
        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        
        bars = axes[2].barh(classes, probs, color=colors, alpha=0.7)
        axes[2].set_xlabel('Probability', fontsize=10)
        axes[2].set_title('Class Probabilities', fontsize=12, fontweight='bold')
        axes[2].set_xlim(0, 1)
        
        for bar, prob in zip(bars, probs):
            axes[2].text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(save_visualization, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_visualization}")
        plt.close()
    
    return result, report


def main():
    """Main function for command-line prediction."""
    parser = argparse.ArgumentParser(description='Predict DR severity')
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='prediction_result.json',
                        help='Output file path')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM visualization')
    parser.add_argument('--report', action='store_true',
                        help='Generate medical report')
    parser.add_argument('--save-visualization', type=str,
                        help='Path to save visualization image')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = load_trained_model(args.model)
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting image: {args.image}")
        
        if args.report:
            result, report = predict_with_report(
                args.image, model, save_visualization=args.save_visualization
            )
            print("\n" + report)
        else:
            result = predict_single_image(args.image, model, return_gradcam=args.gradcam)
            print(f"\nPrediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print("\nAll probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name:20s}: {prob*100:.2f}%")
        
        # Save result
        import json
        with open(args.output, 'w') as f:
            save_result = {k: v for k, v in result.items() 
                          if k not in ['original', 'gradcam']}
            json.dump(save_result, f, indent=2)
        print(f"\n✓ Result saved to: {args.output}")
    
    # Batch prediction
    elif args.folder:
        print(f"\nPredicting all images in: {args.folder}")
        df = batch_predict(args.folder, model, output_csv=args.output)
        print(f"\n✓ Processed {len(df)} images")
    
    else:
        print("Error: Please specify --image or --folder")
        parser.print_help()


if __name__ == "__main__":
    main()
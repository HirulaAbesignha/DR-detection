"""
Gradio web interface for DR detection.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Install with: pip install gradio")
    gr = None

from .utils import CONFIG, load_trained_model
from .predict import predict_single_image
from .report import generate_medical_report


def create_gradio_interface(model):
    """Create Gradio web interface for DR detection."""
    
    def predict_interface(image):
        """Prediction function for Gradio interface."""
        if image is None:
            return "Please upload an image", None, None
        
        try:
            # Make prediction
            result = predict_single_image(image, model, return_gradcam=True)
            
            # Generate medical report
            report = generate_medical_report(result)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(result['original'])
            axes[0].set_title('Original Retinal Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Grad-CAM heatmap
            if 'gradcam' in result:
                axes[1].imshow(result['gradcam'])
                axes[1].set_title('Attention Heatmap (Grad-CAM)', fontsize=12, fontweight='bold')
                axes[1].axis('off')
            
            # Probability distribution
            classes = list(result['all_probabilities'].keys())
            probs = list(result['all_probabilities'].values())
            colors = ['green', 'yellow', 'orange', 'red', 'darkred']
            
            bars = axes[2].barh(classes, probs, color=colors, alpha=0.7)
            axes[2].set_xlabel('Probability', fontsize=10)
            axes[2].set_title('Severity Probabilities', fontsize=12, fontweight='bold')
            axes[2].set_xlim(0, 1)
            
            # Add percentage labels
            for bar, prob in zip(bars, probs):
                axes[2].text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%',
                    va='center',
                    fontsize=9
                )
            
            plt.tight_layout()
            
            # Save and load as image
            plt.savefig('temp_gradio_plot.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Prepare output
            output_dict = {
                'Predicted Class': result['class_name'],
                'Confidence': f"{result['confidence']*100:.2f}%",
                'Probabilities': result['all_probabilities']
            }
            
            return report, 'temp_gradio_plot.png', output_dict
            
        except Exception as e:
            import traceback
            error_msg = f"Error during prediction: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, None, None
    
    # Create interface
    interface = gr.Interface(
        fn=predict_interface,
        inputs=gr.Image(type="numpy", label="Upload Retinal Fundus Image"),
        outputs=[
            gr.Textbox(label="Medical Report", lines=20),
            gr.Image(label="Analysis Visualization", type="filepath"),
            gr.JSON(label="Prediction Details")
        ],
        title="Diabetic Retinopathy Detection System",
        description="""
        **Advanced AI-powered Diabetic Retinopathy Detection**
        
        Upload a retinal fundus image to receive automated severity classification.
        
        **IMPORTANT:** This is an AI screening tool for educational purposes only.
        Always consult a qualified ophthalmologist for diagnosis and treatment.
        """,
        theme="default",
        allow_flagging="never"
    )
    
    return interface


def main():
    """Main function to launch Gradio interface."""
    parser = argparse.ArgumentParser(description='Launch DR detection web interface')
    
    parser.add_argument('--model', type=str, default='./models/best_dr_model.h5',
                        help='Path to trained model')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run server on')
    parser.add_argument('--share', action='store_true',
                        help='Create public shareable link')
    
    args = parser.parse_args()
    
    if gr is None:
        print("Error: Gradio not installed")
        print("Install with: pip install gradio")
        return
    
    # Load model
    print(f"Loading model from: {args.model}")
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("\nPlease train a model first using: python -m src.train")
        return
    
    model = load_trained_model(args.model)
    
    # Create interface
    print("Creating Gradio interface...")
    interface = create_gradio_interface(model)
    
    # Launch
    print(f"\nLaunching web interface on port {args.port}...")
    print(f"Local URL: http://localhost:{args.port}")
    
    if args.share:
        print("Creating public shareable link...")
    
    interface.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
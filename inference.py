"""
Inference script for the trained forgery detection model
Usage: python inference.py --image path/to/image.jpg
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys
from model_optimized import OptimizedForgeryDetector, LightweightForgeryDetector
from handcrafted_optimized import extract_all_handcrafted_features

def load_model(model_path, use_lightweight=False, use_handcrafted=True, device='cuda'):
    """Load the trained model"""
    if use_lightweight:
        model = LightweightForgeryDetector(dropout_rate=0.35, use_handcrafted=use_handcrafted)
    else:
        model = OptimizedForgeryDetector(dropout_rate=0.4, use_handcrafted=use_handcrafted)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device, use_handcrafted=True, target_size=384):
    """
    Predict if an image is authentic or forged
    
    Returns:
        - prediction: 0 = Authentic, 1 = Forged
        - confidence: probability of prediction (0-1)
        - probabilities: dict with probabilities for both classes
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        image_resized = transforms.Resize((target_size, target_size))(image)
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image_resized).unsqueeze(0).to(device)
        
        # Extract handcrafted features if needed
        handcrafted = None
        if use_handcrafted:
            img_array = np.array(image_resized) / 255.0
            handcrafted_feat = extract_all_handcrafted_features(img_array)
            handcrafted = torch.from_numpy(handcrafted_feat).float().unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor, handcrafted)
            probability = torch.sigmoid(output).item()
        
        # Determine class
        prediction = 1 if probability > 0.5 else 0
        
        return {
            'prediction': prediction,
            'prediction_text': 'Forged' if prediction == 1 else 'Authentic',
            'confidence': probability if prediction == 1 else (1 - probability),
            'probability_authentic': 1 - probability,
            'probability_forged': probability,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'prediction': None,
            'prediction_text': None,
            'confidence': None,
            'probability_authentic': None,
            'probability_forged': None,
            'success': False,
            'error': str(e)
        }

def batch_predict(image_dir, model, device, use_handcrafted=True, target_size=384):
    """
    Predict for all images in a directory
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    results = []
    
    print(f"\n📁 Processing images from: {image_dir}")
    print("="*80)
    
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        result = predict_image(image_path, model, device, use_handcrafted, target_size)
        
        if result['success']:
            results.append({
                'filename': image_name,
                **result
            })
            
            status = "✓ Forged" if result['prediction'] == 1 else "✓ Authentic"
            print(f"{status:20} {image_name:40} (conf: {result['confidence']:.4f})")
        else:
            print(f"✗ ERROR {image_name:40} ({result['error']})")
    
    print("="*80)
    
    # Summary statistics
    if results:
        forged_count = sum(1 for r in results if r['prediction'] == 1)
        authentic_count = sum(1 for r in results if r['prediction'] == 0)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n📊 Summary:")
        print(f"   Total images: {len(results)}")
        print(f"   Authentic:    {authentic_count} ({100*authentic_count/len(results):.1f}%)")
        print(f"   Forged:       {forged_count} ({100*forged_count/len(results):.1f}%)")
        print(f"   Avg Confidence: {avg_confidence:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Forgery Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model weights')
    parser.add_argument('--lightweight', action='store_true', help='Use lightweight model')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--target-size', type=int, default=384, help='Input image size')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔥 Using device: {device}")
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    model = load_model(args.model, use_lightweight=args.lightweight, device=device)
    print("✅ Model loaded successfully\n")
    
    # Process input
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            sys.exit(1)
        
        print(f"🖼️  Processing image: {args.image}")
        result = predict_image(args.image, model, device, use_handcrafted=True, target_size=args.target_size)
        
        if result['success']:
            print("\n" + "="*80)
            print(f"📊 PREDICTION RESULT")
            print("="*80)
            print(f"Image:                 {os.path.basename(args.image)}")
            print(f"Prediction:            {result['prediction_text']}")
            print(f"Confidence:            {result['confidence']:.4f}")
            print(f"Probability Authentic: {result['probability_authentic']:.4f}")
            print(f"Probability Forged:    {result['probability_forged']:.4f}")
            print("="*80)
        else:
            print(f"❌ Error: {result['error']}")
    
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
        
        batch_predict(args.dir, model, device, use_handcrafted=True, target_size=args.target_size)
    
    else:
        print("❌ Please provide either --image or --dir argument")
        print("\nUsage examples:")
        print("  Single image:  python inference.py --image path/to/image.jpg")
        print("  Batch:         python inference.py --dir path/to/images/")
        print("  With CPU:      python inference.py --image image.jpg --cpu")
        sys.exit(1)

if __name__ == '__main__':
    main()
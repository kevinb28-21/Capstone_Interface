#!/usr/bin/env python3
"""
Test script to analyze a single image with NDVI/SAVI/GNDVI and ML model
"""
import sys
import json
from pathlib import Path
from image_processor import analyze_crop_health

def test_image(image_path: str, model_path: str = None):
    """Test the complete analysis pipeline on a single image"""
    
    if model_path is None:
        # Try best model first, then health model
        best_model = Path("./models/onion_crop_best_model.h5")
        health_model = Path("./models/onion_crop_health_model.h5")
        
        if best_model.exists():
            model_path = str(best_model)
            print(f"Using model: {model_path}")
        elif health_model.exists():
            model_path = str(health_model)
            print(f"Using model: {model_path}")
        else:
            print("⚠️  No trained model found, running vegetation index analysis only")
            model_path = None
    
    print(f"\n{'='*60}")
    print(f"Testing Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Run complete analysis
    use_tensorflow = model_path is not None
    results = analyze_crop_health(image_path, use_tensorflow=use_tensorflow, model_path=model_path)
    
    # Display results
    print("\n" + "="*60)
    print("VEGETATION INDICES ANALYSIS")
    print("="*60)
    
    if 'ndvi_mean' in results:
        print(f"\n📊 NDVI (Normalized Difference Vegetation Index):")
        print(f"   Mean:   {results['ndvi_mean']:.3f}")
        print(f"   Std:    {results['ndvi_std']:.3f}")
        print(f"   Min:    {results['ndvi_min']:.3f}")
        print(f"   Max:    {results['ndvi_max']:.3f}")
    
    if 'savi_mean' in results:
        print(f"\n📊 SAVI (Soil-Adjusted Vegetation Index):")
        print(f"   Mean:   {results['savi_mean']:.3f}")
        print(f"   Std:    {results['savi_std']:.3f}")
        print(f"   Min:    {results['savi_min']:.3f}")
        print(f"   Max:    {results['savi_max']:.3f}")
    
    if 'gndvi_mean' in results:
        print(f"\n📊 GNDVI (Green Normalized Difference Vegetation Index):")
        print(f"   Mean:   {results['gndvi_mean']:.3f}")
        print(f"   Std:    {results['gndvi_std']:.3f}")
        print(f"   Min:    {results['gndvi_min']:.3f}")
        print(f"   Max:    {results['gndvi_max']:.3f}")
    
    # Health classification from vegetation indices
    if 'health_status' in results:
        print(f"\n🌱 Health Status (from vegetation indices):")
        print(f"   Status: {results['health_status']}")
        print(f"   Score:  {results.get('health_score', 'N/A')}")
        print(f"   Summary: {results.get('summary', 'N/A')}")
    
    # ML Model Results
    if results.get('tensorflow') and results['tensorflow'].get('model_loaded'):
        tf_results = results['tensorflow']
        print(f"\n{'='*60}")
        print("MACHINE LEARNING MODEL ANALYSIS")
        print("="*60)
        print(f"\n🤖 Model: {tf_results.get('model_path', 'N/A')}")
        print(f"   Classification: {tf_results.get('classification', 'N/A')}")
        print(f"   Confidence:     {tf_results.get('confidence', 0):.1%}")
        
        print(f"\n📈 Class Probabilities:")
        all_preds = tf_results.get('all_predictions', {})
        # Sort by probability
        sorted_preds = sorted(all_preds.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_preds:
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {class_name:15s} {prob:6.1%} {bar}")
    else:
        print(f"\n{'='*60}")
        print("MACHINE LEARNING MODEL")
        print("="*60)
        print("   ⚠️  Model not loaded or not available")
        if results.get('tensorflow'):
            print(f"   Reason: {results['tensorflow'].get('error', 'Unknown')}")
    
    print(f"\n{'='*60}")
    print("COMPLETE RESULTS (JSON)")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to a test image if none provided
        test_dir = Path("./training_data/test")
        test_images = []
        
        # Try to find a test image
        for category in ['healthy', 'diseased', 'stressed']:
            cat_dir = test_dir / category
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                if images:
                    test_images.append(str(images[0]))
                    break
        
        if test_images:
            image_path = test_images[0]
            print(f"No image specified, using: {image_path}")
        else:
            print("Usage: python test_single_image.py <image_path> [model_path]")
            print("\nExample:")
            print("  python test_single_image.py ./training_data/test/healthy/image1.jpg")
            sys.exit(1)
    else:
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_image(image_path, model_path)


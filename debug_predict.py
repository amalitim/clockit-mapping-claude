#!/usr/bin/env python3
"""
Debug script to test the prediction functionality
"""

import os
import pandas as pd
import pickle
import traceback
from advanced_classifier import AdvancedTaskTypeClassifier, TaskTypeClassifier
from model_manager import model_manager

def test_model_loading():
    """Test if we can load the model successfully"""
    print("=== Testing Model Loading ===")
    
    # Test 1: Try loading from registry
    try:
        models = model_manager.get_model_list()
        print(f"Found {len(models)} models in registry:")
        for model in models:
            print(f"  - {model['model_id']}: {model['name']} (exists: {model['file_exists']})")
        
        if models:
            latest_model = models[0]
            if latest_model['file_exists']:
                print(f"\nTrying to load model: {latest_model['model_id']}")
                classifier = AdvancedTaskTypeClassifier()
                classifier.load_from_registry(latest_model['model_id'])
                print(f"Successfully loaded model from registry")
                return classifier
    except Exception as e:
        print(f"Failed to load from registry: {e}")
        traceback.print_exc()
    
    # Test 2: Try loading legacy model
    try:
        if os.path.exists('model.pkl'):
            print(f"\nTrying to load legacy model: model.pkl")
            classifier = TaskTypeClassifier()
            classifier.load_model()
            print(f"Successfully loaded legacy model")
            return classifier
    except Exception as e:
        print(f"Failed to load legacy model: {e}")
        traceback.print_exc()
    
    return None

def test_data_loading():
    """Test if we can load a sample file successfully"""
    print("\n=== Testing Data Loading ===")
    
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory doesn't exist: {uploads_dir}")
        return None
    
    files = [f for f in os.listdir(uploads_dir) if f.endswith(('.csv', '.xlsx')) and not f.startswith('predicted_')]
    
    if not files:
        print(f"No test files found in {uploads_dir}")
        return None
    
    test_file = files[0]
    test_path = os.path.join(uploads_dir, test_file)
    print(f"Testing with file: {test_file}")
    
    try:
        if test_file.lower().endswith('.csv'):
            df = pd.read_csv(test_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(test_path, engine='openpyxl')
        
        print(f"Successfully loaded data:")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Sample data:")
        print(df.head(2))
        
        return df, test_file
    
    except Exception as e:
        print(f"Failed to load test file: {e}")
        traceback.print_exc()
        return None

def test_prediction(classifier, df, filename):
    """Test the prediction process"""
    print(f"\n=== Testing Prediction Process ===")
    
    if classifier is None or df is None:
        print("❌ Cannot test prediction - missing classifier or data")
        return False
    
    try:
        print(f"Making predictions on {len(df)} rows...")
        predictions, confidence_scores = classifier.predict(df)
        
        print(f"✅ Predictions successful:")
        print(f"  - Number of predictions: {len(predictions)}")
        print(f"  - Number of confidence scores: {len(confidence_scores)}")
        print(f"  - Sample predictions: {predictions[:5]}")
        print(f"  - Sample confidence scores: {confidence_scores[:5]}")
        
        # Test the response format that the API would return
        df_copy = df.copy()
        df_copy['Predicted_Type'] = predictions
        df_copy['Confidence'] = confidence_scores
        
        # Convert to JSON-serializable format (like in the API)
        predictions_data = []
        for i, row in df_copy.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.int64, np.int32)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.float64, np.float32)):
                    row_dict[col] = float(val)
                else:
                    row_dict[col] = str(val)
            predictions_data.append(row_dict)
        
        print(f"✅ JSON serialization successful: {len(predictions_data)} records")
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        traceback.print_exc()
        return False

def test_api_simulation():
    """Simulate the /predict_file API call"""
    print(f"\n=== Simulating API Call ===")
    
    # Import required modules
    import numpy as np
    from flask import Flask
    
    try:
        # Test the load_model_if_available function
        print("Testing load_model_if_available()...")
        
        # Create a simple classifier instance
        global classifier
        classifier = TaskTypeClassifier()
        
        # Try to load model
        if not classifier.is_trained:
            try:
                # Try to load the latest model from registry
                models = model_manager.get_model_list()
                if models:
                    latest_model = models[0]  # Most recent
                    if latest_model['file_exists']:
                        classifier.load_from_registry(latest_model['model_id'])
                        print("✅ Loaded model from registry")
                    else:
                        print("❌ Latest model file doesn't exist")
                else:
                    print("❌ No models in registry")
                
                # Fallback to legacy model file
                if not classifier.is_trained and os.path.exists('model.pkl'):
                    print("Falling back to legacy model...")
                    classifier.load_model()
                    print("✅ Loaded legacy model")
                    
            except Exception as e:
                print(f"❌ Could not load existing model: {e}")
                traceback.print_exc()
        
        if classifier.is_trained:
            print("✅ Model is trained and ready")
            
            # Test with sample data
            data_result = test_data_loading()
            if data_result:
                df, filename = data_result
                return test_prediction(classifier, df, filename)
        else:
            print("❌ No trained model available")
            return False
                
    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ClockIt Prediction Debug Tool")
    print("=" * 50)
    
    # Test model loading
    classifier = test_model_loading()
    
    # Test data loading
    data_result = test_data_loading()
    
    # Test prediction
    if data_result:
        df, filename = data_result
        test_prediction(classifier, df, filename)
    
    # Test API simulation
    test_api_simulation()
    
    print("\n" + "=" * 50)
    print("Debug complete")
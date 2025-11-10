#!/usr/bin/env python3
"""
Simple debug script to test the prediction functionality
"""

import os
import pandas as pd
import numpy as np
import traceback
from advanced_classifier import AdvancedTaskTypeClassifier, TaskTypeClassifier
from model_manager import model_manager

def main():
    print("ClockIt Prediction Debug Tool")
    print("=" * 50)
    
    # Step 1: Test model loading
    print("\n1. Testing Model Loading...")
    classifier = None
    
    try:
        # Check model registry
        models = model_manager.get_model_list()
        print(f"Found {len(models)} models in registry")
        
        if models:
            latest_model = models[0]
            print(f"Latest model: {latest_model['model_id']}")
            print(f"File exists: {latest_model['file_exists']}")
            
            if latest_model['file_exists']:
                classifier = AdvancedTaskTypeClassifier()
                classifier.load_from_registry(latest_model['model_id'])
                print("SUCCESS: Loaded model from registry")
        
        # Fallback to legacy model
        if classifier is None or not classifier.is_trained:
            if os.path.exists('model.pkl'):
                print("Trying legacy model...")
                classifier = TaskTypeClassifier()
                classifier.load_model()
                print("SUCCESS: Loaded legacy model")
        
        if classifier is None or not classifier.is_trained:
            print("ERROR: No model could be loaded")
            return False
            
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: Test data loading
    print("\n2. Testing Data Loading...")
    
    try:
        uploads_dir = 'uploads'
        files = [f for f in os.listdir(uploads_dir) if f.endswith(('.csv', '.xlsx')) and not f.startswith('predicted_')]
        
        if not files:
            print("ERROR: No test files found")
            return False
        
        test_file = files[0]
        test_path = os.path.join(uploads_dir, test_file)
        print(f"Testing with: {test_file}")
        
        if test_file.lower().endswith('.csv'):
            df = pd.read_csv(test_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(test_path, engine='openpyxl')
        
        print(f"SUCCESS: Loaded {len(df)} rows with columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Test prediction
    print("\n3. Testing Prediction...")
    
    try:
        print("Making predictions...")
        predictions, confidence_scores = classifier.predict(df)
        
        print(f"SUCCESS: Generated {len(predictions)} predictions")
        print(f"Sample predictions: {predictions[:3]}")
        print(f"Sample confidence: {confidence_scores[:3]}")
        
        # Test JSON serialization (like the API does)
        df_result = df.copy()
        df_result['Predicted_Type'] = predictions
        df_result['Confidence'] = confidence_scores
        
        # Convert to API format
        predictions_data = []
        for i, row in df_result.iterrows():
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
            
            # Only process first few for testing
            if i >= 2:
                break
        
        print(f"SUCCESS: JSON serialization works")
        print(f"Sample JSON record: {predictions_data[0]}")
        
        return True
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nAll tests PASSED - prediction should work!")
    else:
        print("\nTests FAILED - there are issues to fix")
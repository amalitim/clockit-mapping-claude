#!/usr/bin/env python3
"""
Test script to simulate the exact API call that the frontend makes

API Documentation:
- Swagger UI: http://localhost:5000/api/docs
- ReDoc: http://localhost:5000/redoc
- OpenAPI Spec: http://localhost:5000/api/docs/openapi.json

Available API Endpoints:
- Model Management: /api/models, /api/config_presets, /api/train_advanced
- Data Processing: /upload, /upload_training, /api/training_files
- Predictions: /predict_file
- Analytics: /api/feature_importance, /api/classes, /api/model_info
"""

import requests
import json
import time

def test_predict_api():
    """Test the /predict_file endpoint"""
    print("Testing /predict_file API endpoint...")
    
    # First, let's test if the Flask app is running
    try:
        response = requests.get('http://localhost:5000/')
        print(f"Flask app is running (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("ERROR: Flask app is not running. Please start it first with: python enhanced_app.py")
        return False
    
    # Test the prediction endpoint with an existing file
    filename = "1757698900_Brock_Team_2025.09_September.partial.xlsx"
    
    payload = {
        "filename": filename
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    print(f"Making POST request to /predict_file with filename: {filename}")
    
    try:
        start_time = time.time()
        response = requests.post(
            'http://localhost:5000/predict_file',
            headers=headers,
            json=payload,
            timeout=60  # 60 second timeout
        )
        
        duration = time.time() - start_time
        print(f"Response received in {duration:.2f} seconds")
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON structure:")
                print(f"  success: {data.get('success')}")
                print(f"  total_predictions: {data.get('total_predictions')}")
                print(f"  output_filename: {data.get('output_filename')}")
                print(f"  message: {data.get('message')}")
                
                if data.get('success'):
                    predictions = data.get('predictions', [])
                    print(f"  predictions count: {len(predictions)}")
                    if predictions:
                        print(f"  sample prediction: {predictions[0]}")
                    return True
                else:
                    print(f"  ERROR in response: {data.get('message')}")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"ERROR: Could not parse JSON response: {e}")
                print(f"Raw response: {response.text[:500]}...")
                return False
        else:
            print(f"ERROR: HTTP {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out after 60 seconds")
        return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        return False

def test_upload_and_predict():
    """Test the full upload + predict workflow"""
    print("\n" + "="*50)
    print("Testing full upload + predict workflow...")
    
    # Check if we have a test file
    import os
    test_files = [
        "uploads/1757698900_Brock_Team_2025.09_September.partial.xlsx",
        "uploads/test_sample.csv"
    ]
    
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("No test file found. Cannot test upload workflow.")
        return False
    
    print(f"Using test file: {test_file}")
    
    # Test upload first
    with open(test_file, 'rb') as f:
        files = {'file': f}
        
        try:
            upload_response = requests.post(
                'http://localhost:5000/upload',
                files=files,
                timeout=30
            )
            
            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                print(f"Upload successful: {upload_data.get('success')}")
                
                if upload_data.get('success'):
                    server_filename = upload_data.get('filename')
                    print(f"Server filename: {server_filename}")
                    
                    # Now test prediction with this filename
                    return test_predict_with_filename(server_filename)
                else:
                    print(f"Upload failed: {upload_data.get('message')}")
                    return False
            else:
                print(f"Upload failed with status: {upload_response.status_code}")
                return False
                
        except Exception as e:
            print(f"Upload error: {e}")
            return False

def test_predict_with_filename(filename):
    """Test prediction with a specific filename"""
    print(f"\nTesting prediction with filename: {filename}")
    
    payload = {"filename": filename}
    
    try:
        response = requests.post(
            'http://localhost:5000/predict_file',
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("SUCCESS: Prediction completed!")
                print(f"Total predictions: {data.get('total_predictions')}")
                return True
            else:
                print(f"Prediction failed: {data.get('message')}")
                return False
        else:
            print(f"HTTP error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Prediction request error: {e}")
        return False

if __name__ == "__main__":
    print("API Test Tool")
    print("=" * 50)
    
    # Test 1: Direct API call with existing file
    success1 = test_predict_api()
    
    # Test 2: Full upload + predict workflow
    success2 = test_upload_and_predict()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("ALL TESTS PASSED! The API should be working correctly.")
    else:
        print("Some tests failed. Check the Flask application logs for details.")
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
import pandas as pd
import numpy as np
import os
import json
import locale
import time
from werkzeug.utils import secure_filename
from advanced_classifier import AdvancedTaskTypeClassifier, TaskTypeClassifier
from model_manager import ModelManager, ModelConfig, model_manager
import tempfile
from io import StringIO, BytesIO

# Set locale to ensure consistent decimal formatting (use period as decimal separator)
try:
    locale.setlocale(locale.LC_NUMERIC, 'C')  # Use C locale for consistent numeric formatting
except locale.Error:
    pass  # If C locale not available, continue with default

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global classifier instance (for backward compatibility)
classifier = TaskTypeClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_if_available():
    """Try to load a model if available"""
    global classifier
    if not classifier.is_trained:
        try:
            # Try to load the latest model from registry
            models = model_manager.get_model_list()
            if models:
                latest_model = models[0]  # Most recent
                if latest_model['file_exists']:
                    classifier.load_from_registry(latest_model['model_id'])
                    return True
            
            # Fallback to legacy model file
            if os.path.exists('model.pkl'):
                print("Loading legacy model...")
                classifier.load_model()
                return True
                
        except Exception as e:
            print(f"Could not load existing model: {e}")
    return classifier.is_trained

@app.route('/')
def index():
    """Main page with enhanced model management"""
    return render_template('enhanced_index.html')

# ==================== ENHANCED MODEL MANAGEMENT ENDPOINTS ====================

@app.route('/api/models')
def api_models():
    """Get list of all registered models"""
    try:
        models = model_manager.get_model_list()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting models: {str(e)}'
        })

@app.route('/api/models/<model_id>')
def api_model_detail(model_id):
    """Get detailed information about a specific model"""
    try:
        metadata = model_manager.get_model_metadata(model_id)
        if not metadata:
            return jsonify({
                'success': False,
                'message': 'Model not found'
            }), 404
        
        return jsonify({
            'success': True,
            'model': metadata.to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting model details: {str(e)}'
        })

@app.route('/api/config_presets')
def api_config_presets():
    """Get available configuration presets"""
    try:
        presets = model_manager.get_config_presets()
        preset_info = {}
        
        for name, config in presets.items():
            preset_info[name] = {
                'name': name,
                'config': config.to_dict(),
                'description': _get_preset_description(name)
            }
        
        return jsonify({
            'success': True,
            'presets': preset_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting presets: {str(e)}'
        })

def _get_preset_description(preset_name):
    """Get description for configuration presets"""
    descriptions = {
        'baseline': 'Original 500-tree configuration (93.2% accuracy, ~60MB)',
        'optimized': 'Balanced 750-tree configuration (93.4% accuracy, ~70MB)',
        'enhanced': 'High-performance with log_loss criterion (97.1% accuracy, ~105MB)',
        'fast': 'Quick training configuration (estimated 92% accuracy, ~45MB)',
        'ultra_high': 'Maximum performance configuration (estimated 97%+ accuracy, ~120MB)'
    }
    return descriptions.get(preset_name, 'Custom configuration')

@app.route('/api/train_advanced', methods=['POST'])
def api_train_advanced():
    """Advanced training endpoint with configuration management"""
    try:
        data = request.get_json()
        
        # Get configuration
        config_source = data.get('config_source', 'preset')
        
        if config_source == 'preset':
            preset_name = data.get('preset_name', 'enhanced')
            classifier_instance = AdvancedTaskTypeClassifier.from_config_preset(preset_name)
        elif config_source == 'custom':
            config_dict = data.get('config', {})
            config = ModelConfig(**config_dict)
            classifier_instance = AdvancedTaskTypeClassifier(config=config)
        elif config_source == 'model':
            base_model_id = data.get('base_model_id')
            base_config = model_manager.get_model_config(base_model_id)
            if not base_config:
                return jsonify({
                    'success': False,
                    'message': f'Base model {base_model_id} not found'
                })
            classifier_instance = AdvancedTaskTypeClassifier(config=base_config)
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid config_source. Use: preset, custom, or model'
            })
        
        # Training parameters
        model_name = data.get('model_name', 'custom_model')
        description = data.get('description', '')
        tags = data.get('tags', [])
        training_data_path = data.get('training_data_path', 'training-data')
        
        # Train the model
        start_time = time.time()
        model_id, performance_metrics = classifier_instance.train(
            training_data_path=training_data_path,
            model_name=model_name,
            description=description,
            tags=tags,
            save_model=True
        )
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'performance': performance_metrics,
            'training_duration': time.time() - start_time,
            'message': f'Model trained successfully! ID: {model_id}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """Load a specific model for predictions"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                'success': False,
                'message': 'model_id is required'
            })
        
        global classifier
        classifier = AdvancedTaskTypeClassifier()
        classifier.load_from_registry(model_id)
        
        metadata = model_manager.get_model_metadata(model_id)
        
        return jsonify({
            'success': True,
            'message': f'Model {model_id} loaded successfully',
            'model_info': {
                'model_id': model_id,
                'name': metadata.name,
                'performance': metadata.performance_metrics,
                'training_date': metadata.training_date
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading model: {str(e)}'
        })

@app.route('/api/delete_model/<model_id>', methods=['DELETE'])
def api_delete_model(model_id):
    """Delete a model from the registry"""
    try:
        success = model_manager.delete_model(model_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_id} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting model: {str(e)}'
        })

# ==================== LEGACY COMPATIBILITY ENDPOINTS ====================

@app.route('/train', methods=['POST'])
def train_model():
    """Legacy training endpoint for backward compatibility"""
    try:
        global classifier
        classifier = TaskTypeClassifier()  # Reset classifier
        train_score, test_score = classifier.train()
        classifier.save_model()
        
        return jsonify({
            'success': True,
            'message': f'Model trained successfully! Training accuracy: {train_score:.3f}, Validation accuracy: {test_score:.3f}',
            'train_accuracy': train_score,
            'validation_accuracy': test_score
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/api/training_files')
def api_training_files():
    """Get information about current training files"""
    try:
        training_files = []
        training_folder = 'training-data'
        
        if os.path.exists(training_folder):
            for filename in os.listdir(training_folder):
                if filename.endswith(('.csv', '.xlsx')):
                    file_path = os.path.join(training_folder, filename)
                    file_type = 'csv' if filename.endswith('.csv') else 'excel'
                    
                    # Get row count
                    try:
                        if filename.endswith('.csv'):
                            df = pd.read_csv(file_path, encoding='utf-8-sig')
                        else:
                            df = pd.read_excel(file_path, engine='openpyxl')
                        row_count = len(df)
                        
                        # Get file stats
                        stat = os.stat(file_path)
                        file_size = stat.st_size
                        
                        training_files.append({
                            'filename': filename,
                            'type': file_type,
                            'rows': row_count,
                            'size': file_size,
                            'size_mb': round(file_size / (1024 * 1024), 2)
                        })
                    except Exception as e:
                        training_files.append({
                            'filename': filename,
                            'type': file_type,
                            'rows': 'Error',
                            'error': str(e)
                        })
        
        return jsonify({
            'success': True,
            'training_files': training_files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting training files: {str(e)}'
        })

# ==================== PREDICTION ENDPOINTS ====================

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True, 
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    else:
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload CSV or XLSX files.'})

@app.route('/predict_file', methods=['POST'])
def predict_file():
    """Make predictions on uploaded file"""
    try:
        # Load model if not loaded
        if not load_model_if_available():
            return jsonify({
                'success': False,
                'message': 'No trained model available. Please train a model first.'
            })
        
        filename = request.json.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'Filename is required'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})
        
        # Load the data
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        else:
            df = pd.read_excel(filepath, engine='openpyxl')
        
        # Make predictions
        predictions, confidence_scores = classifier.predict(df)
        
        # Add predictions to dataframe
        df['Predicted_Type'] = predictions
        df['Confidence'] = confidence_scores
        
        # Save predictions
        output_filename = f"predicted_claude_{filename}"
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        if filename.lower().endswith('.csv'):
            df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        else:
            df.to_excel(output_filepath, index=False, engine='openpyxl')
        
        # Convert to JSON-serializable format for response
        predictions_data = []
        for i, row in df.iterrows():
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
        
        return jsonify({
            'success': True,
            'predictions': predictions_data,
            'output_filename': output_filename,
            'total_predictions': len(predictions_data),
            'message': f'Predictions completed for {len(predictions_data)} records'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error making predictions: {str(e)}'
        })

@app.route('/download/<filename>')
def download_file(filename):
    """Download predicted files"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500

# ==================== VISUALIZATION ENDPOINTS ====================

@app.route('/visualize')
def visualize():
    """Visualization page"""
    return render_template('visualize.html')

@app.route('/api/feature_importance')
def api_feature_importance():
    """Get feature importance for visualization"""
    try:
        if not load_model_if_available():
            return jsonify({
                'success': False,
                'message': 'No trained model available'
            })
        
        importance = classifier.get_feature_importance(top_n=15)
        return jsonify({
            'success': True,
            'feature_importance': importance
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting feature importance: {str(e)}'
        })

# Initialize the application
def initialize_app():
    """Initialize the application"""
    print("🌟 Enhanced Task Type Classifier")
    print("Initializing model manager...")
    
    # Try to load an existing model
    load_model_if_available()
    
    if classifier.is_trained:
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No trained model found. Please train a model through the web interface.")

if __name__ == '__main__':
    initialize_app()
    print("Starting Flask application...")
    print("Open your web browser and navigate to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
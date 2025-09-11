from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
import pandas as pd
import numpy as np
import os
import json
import locale
from werkzeug.utils import secure_filename
from classifier import TaskTypeClassifier
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

# Global classifier instance
classifier = TaskTypeClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def train_model_if_needed():
    """Train the model if not already trained"""
    global classifier
    if not classifier.is_trained:
        try:
            if os.path.exists('model.pkl'):
                print("Loading existing model...")
                classifier.load_model()
            else:
                print("Training new model...")
                classifier.train()
                classifier.save_model()
        except Exception as e:
            print(f"Error training/loading model: {e}")
            return False
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train or retrain the model"""
    try:
        global classifier
        classifier = TaskTypeClassifier()  # Reset classifier
        train_score, test_score = classifier.train()
        classifier.save_model()
        
        return jsonify({
            'success': True,
            'message': f'Model trained successfully! Training accuracy: {train_score:.3f}, Validation accuracy: {test_score:.3f}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/visualize')
def visualize():
    """Visualize word-to-class relationships"""
    if not train_model_if_needed():
        flash('Error: Could not load or train model', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get feature importance
        feature_importance = classifier.get_feature_importance(top_n=15)
        
        # Get class distribution
        classes = classifier.get_class_distribution()
        
        return render_template('visualize.html', 
                             feature_importance=feature_importance,
                             classes=classes)
    except Exception as e:
        flash(f'Error generating visualization: {str(e)}', 'error')
        return redirect(url_for('index'))

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
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and validate the file (handle BOM encoding for CSV)
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            # Check required columns (excluding Type and Source.Name - Source.Name no longer expected in prediction files)
            required_cols = ['Employees', 'Task Name', 'Category', 'Project', 'Billability Status']
            available_cols = df.columns.tolist()
            
            # Store file info in session-like manner (simplified for demo)
            file_info = {
                'filename': filename,
                'filepath': filepath,
                'columns': available_cols,
                'row_count': len(df),
                'has_type': 'Type' in df.columns
            }
            
            return jsonify({
                'success': True,
                'message': f'File uploaded successfully! {len(df)} rows found.',
                'file_info': file_info
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload a CSV or Excel file.'})

@app.route('/predict_file', methods=['POST'])
def predict_file():
    """Make predictions on uploaded file"""
    if not train_model_if_needed():
        return jsonify({'success': False, 'message': 'Could not load or train model'})
    
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'message': 'No filename provided'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})
        
        # Load the file (handle BOM encoding for CSV)
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        # Make predictions
        predictions, confidence_scores = classifier.predict(df)
        
        # Add predictions to dataframe
        df['Predicted_Type'] = predictions
        df['Confidence'] = confidence_scores
        
        # Save the predictions back to the file in original format
        if filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            df.to_csv(filepath, index=False, decimal='.')
        
        # Convert to records for JSON serialization
        records = df.to_dict('records')
        
        # Clean records for JSON serialization - handle NaN values
        import json
        clean_records = []
        for record in records:
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean_record[key] = None
                elif isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
                    clean_record[key] = str(value) if not pd.isna(value) else None
                else:
                    clean_record[key] = value
            clean_records.append(clean_record)
        
        return jsonify({
            'success': True,
            'predictions': clean_records,
            'total_rows': len(df)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error making predictions: {str(e)}'})

@app.route('/update_prediction', methods=['POST'])
def update_prediction():
    """Update a single prediction"""
    try:
        data = request.json
        filename = data.get('filename')
        row_index = data.get('row_index')
        new_type = data.get('new_type')
        
        if not all([filename, row_index is not None, new_type]):
            return jsonify({'success': False, 'message': 'Missing required data'})
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})
        
        # Load file, update prediction, and save
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        if 'Predicted_Type' in df.columns and row_index < len(df):
            df.loc[row_index, 'Predicted_Type'] = new_type
            # Save in the same format as original
            if filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            else:
                df.to_csv(filepath, index=False)
            
            return jsonify({'success': True, 'message': 'Prediction updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Invalid row index or missing predictions'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error updating prediction: {str(e)}'})

@app.route('/download/<filename>')
def download_file(filename):
    """Download the file with predictions"""
    try:
        # Secure the filename
        filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        # Load file and rename prediction column to Type
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
        if 'Predicted_Type' in df.columns:
            # Rename prediction column and keep confidence column
            df['Predicted_Type'] = df['Predicted_Type']  # Keep as Predicted_Type
            # Keep the Confidence column in the final export
            # No need to drop anything - keep both Predicted_Type and Confidence
            
            # Create file content in memory (keep original format)
            if filename.endswith('.xlsx'):
                # For Excel files, keep as Excel
                output = BytesIO()
                # Ensure decimal formatting uses periods
                df.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                output_filename = f"predicted_claude_{filename}"
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                content = output.getvalue()
            else:
                # For CSV files, keep as CSV
                output = StringIO()
                # Ensure decimal separator is period (not comma)
                df.to_csv(output, index=False, decimal='.')
                output.seek(0)
                output_filename = f"predicted_claude_{filename}"
                mimetype = 'text/csv'
                content = output.getvalue()
            
            return Response(
                content,
                mimetype=mimetype,
                headers={
                    'Content-Disposition': f'attachment; filename="{output_filename}"'
                }
            )
        else:
            return jsonify({'error': 'No predictions found in file'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/api/feature_importance')
def api_feature_importance():
    """API endpoint for feature importance data"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        feature_importance = classifier.get_feature_importance(top_n=20)
        return jsonify(feature_importance)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/classes')
def api_classes():
    """API endpoint to get all available task type classes"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        classes = classifier.get_class_distribution()
        
        # Convert numpy array to list if needed and check if empty
        if hasattr(classes, 'tolist'):
            classes = classes.tolist()
        
        if not classes:
            # Fallback: read directly from training data
            training_file = 'training-data/mapped_sample_2025.07.31.csv'
            if os.path.exists(training_file):
                df = pd.read_csv(training_file, encoding='utf-8-sig')
                if 'Type' in df.columns:
                    classes = sorted(df['Type'].unique().tolist())
        
        return jsonify({'success': True, 'classes': classes})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Ensure model is trained on startup
    print("Initializing classifier...")
    train_model_if_needed()
    
    print("Starting Flask application...")
    print("Open your web browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
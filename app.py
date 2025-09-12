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
                    except:
                        row_count = 'Unknown'
                    
                    training_files.append({
                        'name': filename,
                        'type': file_type,
                        'rows': row_count
                    })
        
        return jsonify({
            'success': True,
            'files': training_files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_training', methods=['POST'])
def upload_training_file():
    """Upload a new training file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            name, ext = os.path.splitext(filename)
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{name}_{timestamp}{ext}"
            
            filepath = os.path.join('training-data', filename)
            
            # Ensure training-data directory exists
            os.makedirs('training-data', exist_ok=True)
            
            file.save(filepath)
            
            # Validate the file by trying to load it
            if filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath, engine='openpyxl')
            else:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            # Check required columns
            required_cols = ['Employees', 'Task Name', 'Category', 'Project', 'Billability Status', 'Type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Clean up uploaded file if validation fails
                os.remove(filepath)
                return jsonify({
                    'success': False, 
                    'message': f'Training file is missing required columns: {", ".join(missing_cols)}'
                })
            
            return jsonify({
                'success': True,
                'message': f'Training file uploaded successfully! {len(df)} rows loaded from {filename}.',
                'filename': filename,
                'rows': len(df)
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing training file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload a CSV or Excel file.'})

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
        
        # Clean records for JSON serialization - handle NaN values and all date/time types
        import json
        from datetime import datetime, date, time
        
        clean_records = []
        for record in records:
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean_record[key] = None
                elif isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
                    clean_record[key] = str(value) if not pd.isna(value) else None
                elif isinstance(value, (datetime, date, time)):
                    # Handle Python built-in date/time objects
                    clean_record[key] = str(value)
                elif hasattr(value, 'isoformat'):
                    # Handle any object with isoformat method (datetime-like)
                    clean_record[key] = value.isoformat()
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
            training_files = [
                'training-data/Brock_Team_2024.12_to_2025.08_mapped.xlsx',
                'training-data/mapped_sample_2025.07.31.csv'
            ]
            
            for training_file in training_files:
                if os.path.exists(training_file):
                    if training_file.endswith('.xlsx'):
                        df = pd.read_excel(training_file, engine='openpyxl')
                    else:
                        df = pd.read_csv(training_file, encoding='utf-8-sig')
                    
                    if 'Type' in df.columns:
                        classes = sorted(df['Type'].unique().tolist())
                        break
        
        return jsonify({'success': True, 'classes': classes})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/feature_grid')
def api_feature_grid():
    """API endpoint for feature importance grid (words vs classes)"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        # Get feature importance for all classes
        feature_importance = classifier.get_feature_importance(top_n=25)
        
        # Create a comprehensive word list
        all_words = set()
        for class_features in feature_importance.values():
            for word, score in class_features:
                all_words.add(word)
        
        # Create grid data
        grid_data = []
        classes = list(feature_importance.keys())
        
        for word in sorted(all_words):
            row = {'word': word}
            for class_name in classes:
                # Find score for this word in this class
                score = 0
                for feature_word, feature_score in feature_importance[class_name]:
                    if feature_word == word:
                        score = feature_score
                        break
                row[class_name] = score
            grid_data.append(row)
        
        # Sort by maximum score across all classes
        grid_data.sort(key=lambda x: max(x[class_name] for class_name in classes), reverse=True)
        
        return jsonify({
            'success': True,
            'grid_data': grid_data[:20],  # Top 20 words
            'classes': classes
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/word_frequencies')
def api_word_frequencies():
    """API endpoint for word frequency analysis"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        # Load training data to analyze word frequencies
        training_folder = 'training-data'
        data_frames = []
        
        if os.path.exists(training_folder):
            for filename in os.listdir(training_folder):
                if filename.endswith(('.csv', '.xlsx')):
                    file_path = os.path.join(training_folder, filename)
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    data_frames.append(df)
        
        if not data_frames:
            return jsonify({'error': 'No training data found'})
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Extract text for frequency analysis - ONLY from Task Name column
        task_descriptions = []
        if 'Task Name' in combined_df.columns:
            task_descriptions = combined_df['Task Name'].fillna('').astype(str).tolist()
        
        # Simple word frequency counting
        from collections import Counter
        import re
        
        # Combine all task descriptions and extract words
        combined_text = ' '.join(task_descriptions).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)  # Words with 3+ letters
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
                     'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
                     'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_freq_counter = Counter(filtered_words)
        
        # Get ALL words, not just top 50
        word_freq = word_freq_counter.most_common()  # Get all words
        total_words = sum(word_freq_counter.values())
        
        # Calculate percentages and create response data
        word_data = []
        for word, freq in word_freq:
            percentage = (freq / total_words) * 100
            word_data.append({
                'word': word,
                'frequency': freq,
                'percentage': round(percentage, 4)
            })
        
        return jsonify({
            'success': True,
            'word_frequencies': word_data,
            'total_words': total_words,
            'unique_words': len(word_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/categorical_analysis')
def api_categorical_analysis():
    """API endpoint for categorical data analysis (employees, categories, projects, duration)"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        # Load training data for categorical analysis
        training_folder = 'training-data'
        data_frames = []
        
        if os.path.exists(training_folder):
            for filename in os.listdir(training_folder):
                if filename.endswith(('.csv', '.xlsx')):
                    file_path = os.path.join(training_folder, filename)
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    data_frames.append(df)
        
        if not data_frames:
            return jsonify({'error': 'No training data found'})
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        categorical_data = {}
        
        # Analyze Employees
        if 'Employees' in combined_df.columns:
            employee_counts = combined_df['Employees'].value_counts().head(20)
            categorical_data['employees'] = [
                {'name': str(emp), 'count': int(count)} 
                for emp, count in employee_counts.items()
            ]
        
        # Analyze Categories
        if 'Category' in combined_df.columns:
            category_counts = combined_df['Category'].value_counts()
            categorical_data['categories'] = [
                {'name': str(cat), 'count': int(count)} 
                for cat, count in category_counts.items()
            ]
        
        # Analyze Projects
        if 'Project' in combined_df.columns:
            project_counts = combined_df['Project'].value_counts().head(15)
            categorical_data['projects'] = [
                {'name': str(proj), 'count': int(count)} 
                for proj, count in project_counts.items()
            ]
        
        # Analyze Task Types
        if 'Type' in combined_df.columns:
            type_counts = combined_df['Type'].value_counts()
            categorical_data['task_types'] = [
                {'name': str(task_type), 'count': int(count)} 
                for task_type, count in type_counts.items()
            ]
        
        # Analyze Duration if available
        duration_stats = {}
        for duration_col in ['Duration (decimal)', 'Duration(h)']:
            if duration_col in combined_df.columns:
                duration_data = pd.to_numeric(combined_df[duration_col], errors='coerce').dropna()
                if len(duration_data) > 0:
                    duration_stats[duration_col] = {
                        'total_hours': float(duration_data.sum()),
                        'avg_hours': float(duration_data.mean()),
                        'median_hours': float(duration_data.median()),
                        'min_hours': float(duration_data.min()),
                        'max_hours': float(duration_data.max()),
                        'std_hours': float(duration_data.std()),
                        'count': int(len(duration_data))
                    }
                    break
        
        if duration_stats:
            categorical_data['duration_stats'] = duration_stats
        
        # General statistics
        categorical_data['total_records'] = len(combined_df)
        categorical_data['date_range'] = {
            'total_records': len(combined_df)
        }
        
        # Add date range if date columns exist
        date_columns = ['Date', 'Start Date', 'End Date', 'Created Date']
        for date_col in date_columns:
            if date_col in combined_df.columns:
                try:
                    dates = pd.to_datetime(combined_df[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        categorical_data['date_range'].update({
                            'start_date': dates.min().strftime('%Y-%m-%d'),
                            'end_date': dates.max().strftime('%Y-%m-%d'),
                            'total_days': (dates.max() - dates.min()).days
                        })
                        break
                except:
                    continue
        
        return jsonify({
            'success': True,
            'categorical_data': categorical_data
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information and statistics"""
    if not train_model_if_needed():
        return jsonify({'error': 'Could not load or train model'})
    
    try:
        model_info = {
            'algorithm': 'Random Forest Classifier',
            'description': 'An ensemble learning method that uses multiple decision trees and voting for classification',
            'parameters': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'features': {
                'text_vectorization': 'TF-IDF (Term Frequency-Inverse Document Frequency)',
                'max_features': 1500,
                'ngram_range': '(1, 2) - unigrams and bigrams',
                'stop_words': 'English stop words removed'
            },
            'preprocessing': [
                'Text fields combined: Employees, Task Name, Category, Project, Billability Status',
                'Duration fields converted to numeric (if present)',
                'TF-IDF vectorization with sublinear scaling',
                'Labels encoded using LabelEncoder'
            ],
            'classes': classifier.get_class_distribution().tolist() if hasattr(classifier.get_class_distribution(), 'tolist') else classifier.get_class_distribution(),
            'n_features': len(classifier.data_processor.feature_names) if hasattr(classifier.data_processor, 'feature_names') else 'Unknown'
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Ensure model is trained on startup
    print("Initializing classifier...")
    train_model_if_needed()
    
    print("Starting Flask application...")
    print("Open your web browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
# Task Type Classifier

A fully local Windows application for task type classification and review. This application trains an improved multi-class Random Forest classifier to predict task types from CSV and Excel data and provides a web interface for visualization and prediction.

## Features

- **Local Operation**: Runs entirely offline on Windows 10/11
- **Improved Machine Learning**: Uses optimized Random Forest with class balancing and enhanced TF-IDF features
- **Web Interface**: Simple browser-based UI for all operations
- **Multi-Format Support**: Upload CSV or Excel (.xlsx) files, download in original format
- **Enhanced Interactive Review**: 
  - Edit predictions inline before downloading
  - **Column Sorting**: Sort predictions by any column (ascending/descending)
  - **Real-time Search/Filter**: Filter records across all data fields
  - **Complete Task Type Options**: Edit dropdown shows all 14+ available task types from training data
  - **Reset Functionality**: Clear all filters and return to original view
- **Feature Analysis**: Visualize word-to-class relationships with class-specific insights
- **Robust File Handling**: Supports CSV (various encodings) and Excel (.xlsx) files, handles JSON serialization issues
- **Format Preservation**: Excel files remain as Excel, CSV files remain as CSV throughout the workflow

## Quick Start

### First Time Setup
1. **Install uv** (if not already installed): Visit [uv installation guide](https://github.com/astral-sh/uv)

2. **Create virtual environment and install dependencies**:
   ```bash
   cd clockit-mapping-claude
   uv venv
   uv pip install -r requirements.txt
   ```

### Starting the Application

**Option 1 - Easy Start (Recommended):**
```bash
# Double-click this file in Windows Explorer
run.bat
```

**Option 2 - Command Line:**
```bash
cd clockit-mapping-claude
uv run python app.py
```

**Option 3 - Manual Environment:**
```bash
cd clockit-mapping-claude
.venv\Scripts\activate
python app.py
```

### Using the Application

1. **Open your browser** and navigate to: `http://localhost:5000`

2. **Train the model** (first time only):
   - Click "Train Now" on the home page
   - Uses training data from `/training-data` folder automatically
   - Model will be saved as `model.pkl` for future use

3. **Upload files for prediction**:
   - **CSV files**: Traditional format, continues to work as before
   - **Excel files (.xlsx)**: New preferred format, maintains formatting through workflow

4. **Stop the application**: Press `Ctrl+C` in the terminal or close the command window

## File Structure

```
clockit-mapping-claude/
├── training-data/          # Training CSV files
│   └── mapped_sample_2025.07.31.csv
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── visualize.html
│   └── predict.html
├── uploads/               # Uploaded files (created automatically)
├── app.py                # Main Flask application
├── classifier.py         # Machine learning classifier
├── data_processor.py     # Data preprocessing utilities
├── requirements.txt      # Python dependencies
├── run.bat              # Windows batch file to start app
└── model.pkl            # Trained model (created after training)
```

## Usage

### 1. Training
- Click "Train Now" on the home page
- The system will use data from `/training-data` folder
- Model accuracy and metrics will be displayed
- Trained model is saved as `model.pkl`

### 2. Feature Analysis
- Navigate to "Visualize" to see word-to-class relationships
- View top features for each task type
- Interactive charts show feature importance

### 3. Making Predictions
- Go to "Predict" page
- Upload a CSV or Excel (.xlsx) file without prediction columns
- **No Source.Name column needed** for prediction files
- Click "Generate Predictions" to get classifications
- **Enhanced Predictions Table Features:**
  - **Column Sorting**: Click any column header to sort ascending/descending
  - **Search/Filter**: Use the search box to filter records across all columns
  - **Reset View**: Click "Reset View" to clear all filters and sorting
  - **Complete Edit Options**: Edit dropdown shows all available task types from training data
- Review and edit predictions inline with confidence scores
- Download the final file with 'Predicted_Type' and 'Confidence' columns added
- **Format preservation**: Excel files download as .xlsx, CSV files as .csv

## Excel Workflow (Recommended)

The application now fully supports Excel files as the preferred format for predictions:

### Advantages of Excel Format:
- **Better data integrity**: Preserves cell formatting and data types
- **No encoding issues**: Eliminates UTF-8/BOM encoding problems
- **Professional format**: Standard business format for data exchange
- **Familiar interface**: Most users are comfortable with Excel

### Excel Workflow Steps:
1. **Prepare your Excel file (.xlsx)**:
   - Include required columns: Employees, Task Name, Category, Project, Billability Status
   - **No Source.Name column needed** (major change from previous versions)
   - Include any additional data columns (dates, duration, etc.)

2. **Upload and predict**:
   - Upload your .xlsx file through the web interface
   - Generate predictions with confidence scores
   - Review and edit predictions inline if needed

3. **Download results**:
   - Download as `predicted_yourfile.xlsx` (maintains Excel format)
   - 'Predicted_Type' column added with predictions
   - 'Confidence' column added with prediction confidence scores
   - All original data and formatting preserved

## File Formats

**Supported formats**: CSV (.csv) and Excel (.xlsx)  
**Preferred format**: Excel (.xlsx) for new prediction files  
**Training data**: Remains in CSV format

### Training Data
Must include these columns:
- `Employees`
- `Task Name`
- `Category`
- `Project`
- `Billability Status`
- `Type` (target column for training)

### Prediction Data
Should include required columns (Source.Name column no longer expected):
- `Employees`
- `Task Name`
- `Category`
- `Project`
- `Billability Status`
- Other columns (Duration, dates, etc.)

**Important Changes**:
- **Source.Name column**: No longer required in prediction files
- **Excel format**: Now preferred for new prediction files (.xlsx)
- **Format preservation**: Files maintain their original format (Excel → Excel, CSV → CSV)
- **Training data**: Continues to use CSV format for compatibility

## Technical Details

- **Framework**: Flask web application with Bootstrap UI
- **ML Algorithm**: Optimized Random Forest with class balancing
- **Text Processing**: Enhanced TF-IDF with bigrams (1-2 word phrases)
- **Feature Engineering**: 1,500 TF-IDF features + numeric duration features
- **Enhanced UI Features**:
  - **Dynamic Sorting**: JavaScript-based column sorting for all data fields
  - **Real-time Filtering**: Client-side search across all prediction data
  - **Dynamic Class Loading**: API endpoint (`/api/classes`) provides complete task type list
  - **Responsive Design**: Bootstrap-based responsive interface
- **File Format Support**: 
  - **CSV**: UTF-8 with BOM handling, traditional format
  - **Excel (.xlsx)**: Full support with openpyxl engine, preferred format
  - **Format Preservation**: Maintains original file type through prediction workflow
- **JSON Handling**: Robust serialization with NaN/null value handling
- **Storage**: Local file system, no database required
- **Backwards Compatibility**: Existing CSV workflows continue to work unchanged

## Model Improvements (Latest Version)

- **Class Balance**: Handles imbalanced datasets with `class_weight='balanced'`
- **Overfitting Prevention**: Limited tree depth and minimum samples per node
- **Enhanced Features**: Bigram support for better context understanding
- **Better Generalization**: Reduced overfitting for more realistic predictions

## Installation Requirements

- Python 3.7+
- **uv package manager** (recommended) - handles virtual environments automatically
- Dependencies: Flask, pandas, scikit-learn, numpy, openpyxl (auto-installed via requirements.txt)

## Troubleshooting

### Common Issues:
1. **"Model not trained" error**: Click "Train Now" button first
2. **File upload fails**: Ensure file has required columns (Employees, Task Name, Category, Project, Billability Status)
   - **CSV files**: Check encoding (UTF-8 preferred)
   - **Excel files**: Ensure .xlsx format (not .xls)
   - **Source.Name**: No longer required for prediction files
3. **File format/encoding errors**: Application handles CSV BOM encoding and Excel files automatically
4. **Download issues**: Files maintain original format (Excel → Excel, CSV → CSV)
5. **Port 5000 in use**: Close other applications using port 5000 or modify port in app.py
6. **Excel file errors**: Ensure openpyxl is installed: `uv pip install openpyxl`

### Restarting After Computer Restart:
1. Navigate to project folder: `clockit-mapping-claude`
2. Double-click `run.bat` OR run `uv run python app.py`
3. Open browser to `http://localhost:5000`
4. Model will load automatically (no need to retrain)

## Classes Found in Training Data

The current training data contains these task types:
- **Reporting** (1,848 samples) - Most common
- **Governance** (850 samples)
- **Projects** (831 samples) 
- **Support** (628 samples)
- **NSP 24-AWS DE Training Project** (481 samples)
- AmaliTech Internal, Training & Learning, Leave, Team admin
- Holiday, Planning & Roadmaps, Brock-Cybersecurity Project
- Brock Accounting ID Maintenance Tool, Boart Longyear Consulting Project

## Performance (Improved Model)

- **Training accuracy**: 80.6% (realistic, not overfitted)
- **Validation accuracy**: 80.5% (good generalization)
- **Cross-validation accuracy**: 71.9% (robust performance)
- **Model Type**: RandomForest with 200 trees, max depth 15
- **Features**: 1,500 TF-IDF features including bigrams

The reduced training accuracy (vs previous 100%) indicates better generalization and more realistic predictions on new data.
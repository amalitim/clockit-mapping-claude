# Task Type Classifier

A comprehensive machine learning application for classifying and analyzing task types from time tracking data. Built with Flask and scikit-learn, featuring an intuitive web interface for training, prediction, and advanced data visualization.

**Latest Update (v2.0.1)**: Enhanced stability with critical bug fixes for model loading compatibility and visualization components.

## 🌟 Key Features

### 🤖 Advanced Machine Learning
- **Random Forest Classifier** with optimized parameters for task classification
- **TF-IDF Vectorization** for text feature extraction from task descriptions  
- **Balanced class handling** for imbalanced datasets
- **Cross-validation** with detailed performance metrics
- **Model persistence** with automatic save/load functionality

### 📊 Comprehensive Visualizations (New!)
- **Feature Importance Grid** - Interactive heatmap showing word importance across task types with hover tooltips
- **Task Description Word Analysis** - Word frequency analysis with include/exclude functionality and localStorage persistence
- **Categorical Data Analysis** - Employee activity, project distribution, category breakdown, and duration statistics
- **Class Analysis** - Detailed breakdown by task type with discriminative features and charts
- **Model Information** - Algorithm details, parameters, and preprocessing transparency

### 🎯 Enhanced Prediction & Review
- **Batch Prediction** - Upload CSV/Excel files for automated classification
- **Interactive Review** - Edit predictions with confidence scores and dynamic task type loading
- **Advanced Table Features** - Sort columns, search/filter records, pagination (25/50/100/250/all)
- **Click-to-Edit** - Click any row to modify predictions
- **Export Functionality** - Download results with predictions and confidence scores

### 📁 Smart Training Data Management
- **Multiple File Support** - CSV and Excel file compatibility with automatic format detection
- **File Upload Interface** - Modern upload interface with validation and progress indication
- **Training Status** - Real-time training progress with accuracy metrics and cross-validation scores
- **Data Validation** - Automatic column validation and comprehensive error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- **uv package manager** (recommended) - [Install uv](https://github.com/astral-sh/uv)

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/amalitim/clockit-mapping-claude.git
cd clockit-mapping-claude

# Create virtual environment and install dependencies (using uv)
uv venv
uv pip install flask pandas numpy scikit-learn openpyxl

# Or using pip
pip install flask pandas numpy scikit-learn openpyxl
```

### Starting the Application

**Option 1 - Easy Start (Windows):**
```bash
# Double-click this file in Windows Explorer
run.bat
```

**Option 2 - Command Line (Recommended):**
```bash
uv run python app.py
```

**Option 3 - Manual Environment:**
```bash
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
python app.py
```

Navigate to `http://localhost:5000` in your web browser.

## 📋 Comprehensive Usage Guide

### 1. Training the Model

#### Prepare Training Data
Ensure your CSV/Excel files have these required columns:
- `Employees` - Employee names/IDs
- `Task Name` - Task descriptions (primary text for analysis)
- `Category` - Task categories
- `Project` - Project names
- `Billability Status` - Billing status
- `Type` - Target classification (what you want to predict)

#### Training Process
1. **Upload Training Files**: 
   - Click "Upload New Training File" 
   - Select your CSV/Excel file
   - Files are automatically validated for required columns
   - Files are saved to `training-data/` folder with timestamps

2. **Train the Model**:
   - Click "Train with Current Files" 
   - View real-time training progress
   - See training accuracy, validation accuracy, and cross-validation scores
   - Model automatically saved as `model.pkl`

3. **View Training Results**:
   - Detailed classification report with precision/recall/F1-score
   - Class distribution analysis
   - Training file information and row counts

### 2. Making Predictions

#### Upload & Predict
1. **Upload Prediction File**: Same format as training data (excluding the `Type` column)
2. **Generate Predictions**: Click "Make Predictions" to classify all tasks
3. **Review Results**: 
   - Interactive table with sorting, filtering, and pagination
   - Confidence scores for each prediction
   - Edit predictions by clicking any row
   - Complete task type dropdown with all available classes

#### Advanced Table Features
- **Sorting**: Click column headers to sort (ascending/descending)
- **Search/Filter**: Real-time search across all columns
- **Pagination**: Choose 25/50/100/250 or view all records
- **Reset**: Return to original view with one click
- **Click-to-Edit**: Click any row to open edit modal

4. **Export Results**: Download file with `Predicted_Type` and `Confidence` columns

### 3. Advanced Data Analysis & Visualization

#### Feature Importance Grid
- **Interactive Heatmap**: Words as rows, task types as columns
- **Hover Tooltips**: Detailed importance scores for each word-type combination
- **Color Coding**: Intensity indicates feature importance
- **Sticky Headers**: Easy navigation through large datasets

#### Task Description Word Analysis
- **Word Frequency Table**: All words from task descriptions with counts and percentages
- **Include/Exclude Functionality**: Toggle individual words in/out of percentage calculations
- **Persistent Exclusions**: Uses localStorage to remember excluded words across sessions
- **Advanced Filtering**: View all words, excluded only, or included only
- **Bulk Operations**: Include/exclude all words with one click
- **Real-time Recalculation**: Percentages update dynamically based on included words

#### Categorical Data Analysis
- **Duration Statistics**: Total, average, median, min/max task hours
- **Employee Activity**: Top contributors with task counts and percentages
- **Project Distribution**: Most active projects with activity breakdown
- **Category Analysis**: Task category distribution with percentages
- **Task Type Overview**: Classification distribution with color-coded badges
- **Dataset Overview**: Total records, date ranges, and time span coverage

#### Class Analysis
- **Task Type Breakdown**: Detailed analysis for each classification
- **Discriminative Features**: Top words that distinguish each task type
- **Interactive Charts**: Doughnut charts showing feature importance distribution
- **Professional Layout**: Card-based design with hover effects

#### Model Information
- **Algorithm Details**: Complete Random Forest configuration and rationale
- **Parameters**: All model hyperparameters with explanations
- **Text Processing**: TF-IDF configuration, n-gram settings, stop words
- **Preprocessing Steps**: Complete data transformation pipeline
- **Feature Statistics**: Number of features and class information

## 🎨 Advanced Features Detail

### Word Frequency Analysis with Smart Exclusion
- **Persistent Exclusions**: Browser localStorage remembers your excluded words
- **Dynamic Percentage Recalculation**: Only included words count toward percentages
- **Visual Indicators**: Excluded rows highlighted in yellow
- **Smart Filtering**: Quickly find excluded or included words
- **Bulk Actions**: Exclude common noise words (meetings, calls, etc.) en masse

### Enhanced Prediction Interface
- **Professional Design**: Modern Bootstrap-based interface with animations
- **Performance Optimized**: Handles large datasets with client-side pagination
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Accessibility**: Proper ARIA labels, keyboard navigation, screen reader support

### Training Data Intelligence
- **Format Auto-Detection**: Automatically handles CSV and Excel files
- **Encoding Handling**: Proper UTF-8 BOM handling for international characters
- **Validation & Feedback**: Clear error messages for missing columns or invalid data
- **Progress Indication**: Visual feedback during file upload and training

## 📊 Data Requirements & Formats

### Training Data Format
| Column | Description | Required | Notes |
|--------|-------------|----------|--------|
| Employees | Employee name/ID | Yes | Used for categorical analysis |
| Task Name | Task description | Yes | **Primary text for word analysis** |
| Category | Task category | Yes | Used for categorical analysis |
| Project | Project name | Yes | Used for categorical analysis |
| Billability Status | Billing status | Yes | Used for classification features |
| Type | Task classification | Yes | **Target variable (training only)** |
| Duration (decimal) | Task duration in hours | Optional | Used for duration statistics |

### Prediction Data Format
Same as training data but **without** the `Type` column (this gets predicted).

**Important**: Word frequency analysis now focuses **only** on the "Task Name" column, providing cleaner insights into actual task content rather than mixing in employee names, categories, and projects.

## 🔧 Technical Architecture

### Machine Learning Pipeline
- **Algorithm**: Random Forest Classifier (200 estimators, max depth 15)
- **Text Processing**: TF-IDF vectorization (1,500 features, unigrams + bigrams)
- **Feature Engineering**: Combined text fields, normalized duration, label encoding
- **Class Balancing**: Weighted classes handle imbalanced datasets
- **Validation**: 5-fold stratified cross-validation with performance metrics

### Web Application Stack
- **Backend**: Flask with RESTful API design
- **Frontend**: Bootstrap 5 with custom CSS and JavaScript
- **Data Processing**: Pandas with NumPy for numerical operations
- **File Handling**: Support for CSV (UTF-8 BOM) and Excel (.xlsx) formats
- **Storage**: Local file system, no database required

### Performance & Security
- **Lazy Loading**: Visualizations load only when needed
- **Client-side Caching**: Improved performance with intelligent caching
- **Local Processing**: All data stays on your machine
- **Error Handling**: Comprehensive error management with user-friendly messages
- **Responsive Design**: Works on all screen sizes and devices

## 📁 Project Structure

```
clockit-mapping-claude/
├── app.py                    # Main Flask application with API endpoints
├── classifier.py             # Random Forest model implementation
├── data_processor.py         # Data preprocessing and feature engineering
├── templates/
│   ├── base.html            # Base template with navigation
│   ├── index.html           # Training interface with file upload
│   ├── predict.html         # Prediction interface with advanced table
│   └── visualize.html       # Comprehensive visualization dashboard
├── training-data/           # Training data files (CSV/Excel)
├── uploads/                 # Uploaded prediction files
├── .venv/                   # Virtual environment (created by uv)
├── model.pkl               # Trained model (created after training)
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows batch file for easy startup
└── README.md              # This comprehensive documentation
```

## 🎯 API Endpoints

The application provides several API endpoints for data access:

- `GET /api/training_files` - List current training files with metadata
- `POST /train` - Train the model with current training data
- `POST /upload_training` - Upload new training files
- `GET /api/feature_importance` - Get feature importance data for visualizations
- `GET /api/feature_grid` - Get feature importance grid (words vs classes)
- `GET /api/word_frequencies` - Get word frequency analysis from task descriptions
- `GET /api/categorical_analysis` - Get categorical data analysis (employees, projects, etc.)
- `GET /api/classes` - Get all available task type classes
- `GET /api/model_info` - Get detailed model information and parameters

## 🚀 Performance Benchmarks

### Model Performance (Current Training Data)
- **Training Accuracy**: ~81.2%
- **Validation Accuracy**: ~78.7% 
- **Cross-Validation**: ~74.5% (±16.1%)
- **Classes**: 14 different task types
- **Training Samples**: ~7,260 records
- **Features**: 1,500 TF-IDF features + duration features

### Application Performance
- **Startup Time**: < 3 seconds
- **Training Time**: 2-5 seconds (depending on data size)
- **Prediction Time**: < 1 second for typical files (100-1000 records)
- **Visualization Loading**: < 2 seconds for all charts and tables
- **Memory Usage**: ~100-200MB (typical dataset)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow the existing code style and patterns
4. **Test thoroughly**: Ensure all features work as expected
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes and their benefits

### Development Guidelines
- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add comments for complex logic
- Test with both CSV and Excel files
- Ensure responsive design for new UI elements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues & Solutions

#### Training Issues
- **"Model not trained" error**: Click "Train with Current Files" button first
- **Training fails**: Ensure training files have all required columns
- **Low accuracy**: Check data quality and ensure sufficient training samples per class

#### File Upload Issues  
- **Upload fails**: Verify file format (CSV or Excel .xlsx only)
- **Missing columns error**: Ensure all required columns are present and named correctly
- **Encoding errors**: Use UTF-8 encoding for CSV files (handled automatically)

#### Prediction Issues
- **No predictions generated**: Ensure model is trained first
- **Edit dropdown empty**: Check that training data loaded successfully
- **Download fails**: Verify predictions were generated successfully

#### Visualization Issues
- **Charts not loading**: Check browser console for JavaScript errors
- **Word exclusions not persisting**: Ensure browser localStorage is enabled
- **Grid not displaying**: Verify model is trained and feature importance is available

#### Performance Issues
- **Slow loading**: Large files may take longer; consider using pagination
- **Memory errors**: For very large datasets, consider splitting files
- **Browser crashes**: Try using smaller page sizes in the prediction table

### Getting Help

1. **Check browser console**: Look for JavaScript errors (F12 in most browsers)
2. **Verify data format**: Ensure your files match the required column structure
3. **Restart application**: Close terminal and run `uv run python app.py` again
4. **Clear browser cache**: Sometimes cached data can cause issues
5. **Check model file**: If issues persist, delete `model.pkl` and retrain

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)  
- **Browser**: Modern browser with JavaScript enabled
- **Disk Space**: 1GB free space for application and data files

## 🎯 Roadmap & Future Enhancements

### Planned Features
- [ ] **Export Training Insights**: Download comprehensive training analysis reports
- [ ] **Model Comparison**: Compare different model configurations and performance
- [ ] **Batch Processing API**: REST API for automated batch predictions
- [ ] **Advanced Filtering**: Custom filter builders for complex data queries
- [ ] **Custom Stop Words**: User-defined stop word lists for word frequency analysis
- [ ] **Performance Tracking**: Monitor model performance over time
- [ ] **Data Quality Metrics**: Automated data quality assessment and recommendations

### Technical Improvements
- [ ] **Database Support**: Optional database backend for large-scale deployments
- [ ] **Multi-language Support**: Interface localization
- [ ] **Advanced Visualizations**: More chart types and interactive features
- [ ] **Mobile App**: Native mobile app for field data collection
- [ ] **Cloud Deployment**: Docker containers and cloud deployment guides

## 🏆 Recognition

This application demonstrates best practices in:
- **Machine Learning Engineering**: Proper model validation, feature engineering, and performance monitoring
- **Web Application Development**: RESTful API design, responsive UI, and user experience
- **Data Science**: Comprehensive data analysis, visualization, and insight generation
- **Software Engineering**: Clean code architecture, error handling, and documentation

---

**Built with ❤️ using Flask, scikit-learn, Bootstrap, and modern web technologies**

**🤖 Enhanced with Claude Code for intelligent task classification and comprehensive data analysis**
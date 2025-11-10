# ClockIt Mapping Claude - Project Planning

## Project Overview
Machine learning-based task type classification system using Flask web application with advanced model management and prediction capabilities.

## Development Environment

### Running the Application

**IMPORTANT: This project is now Dockerized. Always use Docker to run the application:**

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop the application
docker-compose down
```

The application will be available at: http://localhost:5000

### Docker Configuration
- **Port**: 5000 (mapped to host)
- **Volumes**:
  - `./uploads` - User uploaded files
  - `./training-data` - Training datasets
  - `./models` - Trained model files
  - `./model_registry.json` - Model registry

### Why Docker?
- Consistent environment across development and deployment
- Simplified dependency management
- Easy to deploy and scale
- Isolated from host system

## Architecture

### Core Components
1. **Flask Web Application** (`app.py`)
   - Main application server
   - Routes for training, prediction, and visualization
   - Model management API endpoints

2. **Advanced Classifier** (`advanced_classifier.py`)
   - Enhanced machine learning model with configurable parameters
   - Support for multiple model configurations (baseline, optimized, enhanced)
   - Model registry integration

3. **Model Manager** (`model_manager.py`)
   - Model versioning and registry
   - Configuration management
   - Model metadata tracking

4. **Data Processor** (`data_processor.py`)
   - Data preprocessing and feature engineering
   - TF-IDF vectorization
   - Label encoding

## Development Workflow

### Making Changes
1. Start Docker container: `docker-compose up -d`
2. Make code changes
3. Test changes in the running application
4. Commit incrementally to git
5. Rebuild Docker image if needed: `docker-compose build`

### Testing
- Always test changes through the web interface at http://localhost:5000
- Test both training and prediction workflows
- Verify API endpoints work correctly
- Check visualizations render properly

## Key Features
- Task type classification using Random Forest
- Multiple model configuration presets
- Model versioning and registry
- File upload and prediction
- Interactive visualizations
- API documentation (development mode)

## Latest Updates
- **2025-11-10**: Trained Ultra High model with updated data (10,153 samples, 16 classes)
  - Model ID: `UH_Model_2025.10.data_20251110_165457_642fc641`
  - Configuration: 1,500 trees, 6,000 TF-IDF features, 8-fold CV
  - Performance: 91.0% test accuracy, 81.9% CV accuracy (±13.5%)
  - Training time: ~17.5 minutes
  - Best performing classes: Clean Sky Energy, Brock projects (F1: 1.00)
- **2025-09-30**: Added Docker support for easier deployment and consistent environment
- **2025-09-30**: Added debug utilities for prediction verification
- **2025-09-30**: Updated training data to latest version

## Model Performance Summary

### Current Best Model
- **Model**: UH_Model_2025.10.data (Nov 10, 2025)
- **Test Accuracy**: 91.0%
- **Cross-Validation**: 81.9% (±13.5%)
- **Samples**: 10,153 training samples
- **Classes**: 16 task types

### Performance by Class (F1-Score)
- Perfect (1.00): Clean Sky Energy, Brock-Cybersecurity, Boart Longyear Enterprise Reporting
- Excellent (>0.95): NSP 24-AWS DE Training (0.99), Brock Accounting Tool (0.98), Team admin (0.95)
- Strong (>0.90): AmaliTech Internal (0.92), Leave (0.92), Reporting (0.92), Projects (0.90)
- Good (>0.80): Holiday (0.89), Support (0.85), Governance (0.83)
- Moderate: Training & Learning (0.79)

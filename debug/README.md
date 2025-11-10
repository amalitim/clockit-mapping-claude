# Debug Scripts

This folder contains debug and testing scripts used during development. These scripts are kept for reference but are not part of the main application workflow.

## Scripts

### Debugging Tools
- **debug_predict.py** - Debug tool for investigating prediction issues
- **debug_row_mismatch.py** - Diagnoses row count mismatches between predictions
- **simple_debug.py** - Simple debugging utilities
- **check_all_predictions.py** - Validates predictions across the dataset
- **verify_specific_file.py** - Verifies specific file predictions

### Search and Analysis
- **search_monday_holiday.py** - Searches for Monday/Holiday classification patterns
- **search_specific_task.py** - Searches for specific task entries

### Testing
- **test_api.py** - API endpoint testing script

### Utilities
- **register_existing_models.py** - One-time script to register legacy models in the registry

## Usage

These scripts are standalone and can be run directly:
```bash
python debug/script_name.py
```

Note: Some scripts may require the Flask application to be running or may need path adjustments.

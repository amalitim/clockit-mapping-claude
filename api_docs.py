"""
API Documentation module for ClockIt Task Type Classifier
Creates OpenAPI 3.0 specification and provides documentation endpoints
"""

from flask import Flask, jsonify, render_template_string

def get_openapi_spec():
    """Generate OpenAPI 3.0 specification for the API"""

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "ClockIt Task Classifier API",
            "version": "2.0.2",
            "description": """
A comprehensive machine learning API for classifying and analyzing task types from time tracking data.

## Features
- **Advanced Model Management**: Train, load, and manage multiple ML models with versioning
- **Batch Prediction**: Upload and classify large datasets (CSV/Excel)
- **Real-time Analytics**: Feature importance, word frequency, and categorical analysis
- **Configuration Presets**: Pre-configured model settings for different accuracy/speed trade-offs
- **Interactive Training**: Web-based model training with real-time progress tracking

## Model Performance
- **Ultra-High Accuracy**: Up to 94%+ validation accuracy with optimized Random Forest models
- **Fast Prediction**: < 1 second for typical files (100-1000 records)
- **Scalable**: Handles datasets with 10,000+ records efficiently

## Getting Started
1. **Train a Model**: Use `/api/train_advanced` with configuration presets
2. **Upload Data**: Use `/upload` or `/upload_training` to upload CSV/Excel files
3. **Make Predictions**: Use `/predict_file` to classify your data
4. **Analyze Results**: Use visualization endpoints for insights
            """,
            "contact": {
                "name": "ClockIt Support",
                "url": "https://github.com/your-repo/clockit-mapping"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5000",
                "description": "Development server"
            }
        ],
        "tags": [
            {
                "name": "models",
                "description": "Model Management Operations"
            },
            {
                "name": "data",
                "description": "Data Processing Operations"
            },
            {
                "name": "prediction",
                "description": "Prediction Operations"
            },
            {
                "name": "analytics",
                "description": "Analytics and Visualization"
            }
        ],
        "paths": {
            "/api/models": {
                "get": {
                    "tags": ["models"],
                    "summary": "List all registered models",
                    "description": "Get detailed metadata for all registered models including configuration and performance metrics",
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/ModelMetadata"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/models/{model_id}": {
                "get": {
                    "tags": ["models"],
                    "summary": "Get model details",
                    "description": "Get detailed information about a specific model",
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Unique model identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Model details",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ModelMetadata"}
                                }
                            }
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                },
                "delete": {
                    "tags": ["models"],
                    "summary": "Delete model",
                    "description": "Remove a model from the registry",
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Unique model identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Model deleted successfully"
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/config_presets": {
                "get": {
                    "tags": ["models"],
                    "summary": "Get configuration presets",
                    "description": """Get available configuration presets for model training:
- **fast**: Quick training, moderate accuracy (~88%)
- **balanced**: Good balance of speed and accuracy (~90%)
- **enhanced**: High accuracy with reasonable training time (~92%)
- **ultra_high**: Maximum accuracy, longer training time (~94%+)""",
                    "responses": {
                        "200": {
                            "description": "Available presets",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "additionalProperties": {"$ref": "#/components/schemas/ModelConfig"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/train_advanced": {
                "post": {
                    "tags": ["models"],
                    "summary": "Train advanced model",
                    "description": """Train a new model with advanced configuration options.
Supports three configuration types:
1. **Preset**: Use predefined optimized configurations
2. **Custom**: Specify custom parameters
3. **Model-based**: Use existing model as base configuration""",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TrainingRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Training completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TrainingResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid configuration",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/load_model": {
                "post": {
                    "tags": ["models"],
                    "summary": "Load model for predictions",
                    "description": "Load a specific model to use for predictions",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_id": {
                                            "type": "string",
                                            "description": "Model ID to load"
                                        }
                                    },
                                    "required": ["model_id"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Model loaded successfully"
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/upload": {
                "post": {
                    "tags": ["data"],
                    "summary": "Upload prediction file",
                    "description": """Upload a CSV or Excel file for prediction.

Expected columns:
- **Employees**: Employee names
- **Task Name**: Task descriptions
- **Category**: Task categories
- **Project**: Project names
- **Billability Status**: Billable/Non-billable
- **Duration(h)** or **Duration (decimal)**: Time spent

The **Type** column should be excluded (will be predicted).""",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "CSV or Excel file"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "File uploaded successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UploadResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid file format",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/upload_training": {
                "post": {
                    "tags": ["data"],
                    "summary": "Upload training file",
                    "description": "Upload a training file with all columns including **Type** for model training",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "CSV or Excel training file"
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Training file uploaded successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UploadResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid file format",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/training_files": {
                "get": {
                    "tags": ["data"],
                    "summary": "List training files",
                    "description": "Get metadata about all training files including record counts and column information",
                    "responses": {
                        "200": {
                            "description": "Training files metadata",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TrainingFilesResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/predict_file": {
                "post": {
                    "tags": ["prediction"],
                    "summary": "Make predictions",
                    "description": """Generate task type predictions with confidence scores for an uploaded file.

Returns a new file with additional columns:
- **Predicted_Type**: Predicted task type
- **Confidence**: Prediction confidence score (0-1)""",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Predictions generated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PredictionResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "File not found or invalid",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "Model not loaded",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/feature_importance": {
                "get": {
                    "tags": ["analytics"],
                    "summary": "Get feature importance",
                    "description": "Get the most important words/features for each task type class",
                    "responses": {
                        "200": {
                            "description": "Feature importance analysis",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/FeatureImportanceResponse"}
                                }
                            }
                        },
                        "500": {
                            "description": "No model loaded",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/classes": {
                "get": {
                    "tags": ["analytics"],
                    "summary": "Get task type classes",
                    "description": "Get all available task types that the model can predict",
                    "responses": {
                        "200": {
                            "description": "Available task type classes",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "classes": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/model_info": {
                "get": {
                    "tags": ["analytics"],
                    "summary": "Get current model info",
                    "description": "Get comprehensive information about the currently loaded model",
                    "responses": {
                        "200": {
                            "description": "Current model information",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ModelInfo"}
                                }
                            }
                        },
                        "500": {
                            "description": "No model loaded",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "ModelConfig": {
                    "type": "object",
                    "properties": {
                        "n_estimators": {"type": "integer", "description": "Number of trees in Random Forest", "example": 500},
                        "max_depth": {"type": "integer", "nullable": True, "description": "Maximum depth of trees", "example": 20},
                        "criterion": {"type": "string", "description": "Split criterion", "example": "log_loss"},
                        "max_features_tfidf": {"type": "integer", "description": "Maximum TF-IDF features", "example": 3000},
                        "ngram_range": {"type": "array", "items": {"type": "integer"}, "description": "N-gram range", "example": [1, 3]},
                        "min_df": {"type": "integer", "description": "Minimum document frequency", "example": 1},
                        "max_df": {"type": "number", "description": "Maximum document frequency ratio", "example": 0.90},
                        "test_size": {"type": "number", "description": "Test set size ratio", "example": 0.2},
                        "cv_folds": {"type": "integer", "description": "Cross-validation folds", "example": 5},
                        "random_state": {"type": "integer", "description": "Random seed", "example": 42}
                    }
                },
                "PerformanceMetrics": {
                    "type": "object",
                    "properties": {
                        "training_accuracy": {"type": "number", "description": "Training accuracy", "example": 0.98},
                        "validation_accuracy": {"type": "number", "description": "Validation accuracy", "example": 0.94},
                        "cv_accuracy": {"type": "number", "description": "Cross-validation mean accuracy", "example": 0.93},
                        "cv_std": {"type": "number", "description": "Cross-validation std deviation", "example": 0.02},
                        "training_duration": {"type": "number", "description": "Training time in seconds", "example": 45.6}
                    }
                },
                "ModelMetadata": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Model identifier", "example": "enhanced_model_20250929_143022"},
                        "name": {"type": "string", "description": "Model name", "example": "enhanced_model"},
                        "description": {"type": "string", "description": "Model description", "example": "Enhanced model with log_loss criterion"},
                        "training_date": {"type": "string", "description": "Training timestamp", "example": "2025-09-29 14:30:22"},
                        "config": {"$ref": "#/components/schemas/ModelConfig"},
                        "performance_metrics": {"$ref": "#/components/schemas/PerformanceMetrics"},
                        "tags": {"type": "array", "items": {"type": "string"}, "example": ["enhanced", "production"]},
                        "file_size_mb": {"type": "number", "description": "Model file size in MB", "example": 125.4}
                    }
                },
                "TrainingRequest": {
                    "type": "object",
                    "required": ["config_type", "model_name"],
                    "properties": {
                        "config_type": {"type": "string", "enum": ["preset", "custom", "model_based"], "description": "Configuration type"},
                        "preset_name": {"type": "string", "enum": ["fast", "balanced", "enhanced", "ultra_high"], "description": "Preset name"},
                        "custom_config": {"$ref": "#/components/schemas/ModelConfig"},
                        "base_model_id": {"type": "string", "description": "Base model ID"},
                        "model_name": {"type": "string", "description": "New model name", "example": "my_enhanced_model"},
                        "description": {"type": "string", "description": "Model description"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "selected_files": {"type": "array", "items": {"type": "string"}, "description": "Selected training files"}
                    }
                },
                "TrainingResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "model_id": {"type": "string", "description": "Created model ID"},
                        "performance_metrics": {"$ref": "#/components/schemas/PerformanceMetrics"},
                        "message": {"type": "string", "description": "Training completion message"}
                    }
                },
                "PredictionRequest": {
                    "type": "object",
                    "required": ["filename"],
                    "properties": {
                        "filename": {"type": "string", "description": "Uploaded filename", "example": "my_data.xlsx"}
                    }
                },
                "PredictionResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "message": {"type": "string", "example": "Predictions generated successfully"},
                        "total_predictions": {"type": "integer", "example": 150},
                        "output_filename": {"type": "string", "example": "predictions_my_data_20250929_143045.xlsx"},
                        "predictions": {"type": "array", "items": {"type": "object"}}
                    }
                },
                "UploadResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "message": {"type": "string", "example": "File uploaded successfully"},
                        "filename": {"type": "string", "example": "1727620845_my_data.xlsx"},
                        "original_filename": {"type": "string", "example": "my_data.xlsx"},
                        "file_size": {"type": "integer", "example": 524288},
                        "analysis": {"type": "object", "description": "File content analysis"}
                    }
                },
                "TrainingFilesResponse": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "object"}},
                        "total_files": {"type": "integer", "example": 3},
                        "total_records": {"type": "integer", "example": 1500}
                    }
                },
                "FeatureImportanceResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "feature_importance": {"type": "object", "description": "Feature importance by class"},
                        "model_info": {"type": "object", "description": "Current model information"}
                    }
                },
                "ModelInfo": {
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string", "example": "RandomForestClassifier"},
                        "parameters": {"type": "object"},
                        "feature_count": {"type": "integer", "example": 3000},
                        "class_count": {"type": "integer", "example": 12},
                        "training_samples": {"type": "integer", "example": 1200}
                    }
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "error": {"type": "string", "description": "Error message"},
                        "details": {"type": "string", "description": "Error details"}
                    }
                }
            }
        }
    }

    return spec

def initialize_api_documentation(app: Flask):
    """Initialize API documentation routes"""

    @app.route('/api/docs/openapi.json')
    def openapi_spec():
        """Return OpenAPI specification (development only)"""
        if not app.debug:
            return jsonify({'error': 'API documentation disabled in production mode'}), 404
        return jsonify(get_openapi_spec())

    @app.route('/api/docs')
    def swagger_ui():
        """Swagger UI documentation page (development only)"""
        if not app.debug:
            return jsonify({'error': 'API documentation disabled in production mode'}), 404
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>ClockIt Task Classifier API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css" />
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin:0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: '/api/docs/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                // Security: Disable API testing functionality
                tryItOutEnabled: false,
                supportedSubmitMethods: [],
                onComplete: function() {
                    // Hide all "Try it out" buttons as additional security
                    const tryButtons = document.querySelectorAll('.try-out__btn');
                    tryButtons.forEach(btn => btn.style.display = 'none');

                    // Add security notice
                    const headerEl = document.querySelector('.info');
                    if (headerEl) {
                        const notice = document.createElement('div');
                        notice.className = 'info-description';
                        notice.innerHTML = '<p><strong>🔒 Security Notice:</strong> API testing is disabled for security. This is a read-only documentation view.</p>';
                        notice.style.cssText = 'background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 10px 0;';
                        headerEl.appendChild(notice);
                    }
                }
            });
        };
    </script>
</body>
</html>
        ''')

    return app
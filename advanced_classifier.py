import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from typing import Dict, Optional, Tuple

from data_processor import DataProcessor
from model_manager import ModelManager, ModelConfig, model_manager


class AdvancedTaskTypeClassifier:
    """Enhanced classifier with configuration management and model versioning"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()  # Use default config if none provided
        self.data_processor = DataProcessor()
        self.model = None
        self.is_trained = False
        self.model_manager = model_manager
        self._build_model()
    
    def _build_model(self):
        """Build model from configuration"""
        # Update data processor with config
        self.data_processor.update_config(
            max_features=self.config.max_features_tfidf,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df
        )
        
        # Build Random Forest model
        model_params = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'min_samples_leaf': self.config.min_samples_leaf,
            'max_features': self.config.max_features,
            'criterion': self.config.criterion,
            'bootstrap': self.config.bootstrap,
            'oob_score': self.config.oob_score,
            'class_weight': self.config.class_weight,
            'random_state': self.config.random_state,
            'n_jobs': -1,
            'verbose': 1
        }
        
        # Add max_samples if specified
        if self.config.max_samples is not None:
            model_params['max_samples'] = self.config.max_samples
        
        self.model = RandomForestClassifier(**model_params)
    
    def train(self, 
              training_data_path: str = 'training-data',
              model_name: str = 'task_classifier',
              description: str = '',
              tags: list = None,
              save_model: bool = True) -> Tuple[str, Dict]:
        """Train the classifier and register it with the model manager"""
        
        start_time = time.time()
        
        print(f"🚀 Starting training with configuration:")
        print(f"   Trees: {self.config.n_estimators}")
        print(f"   Criterion: {self.config.criterion}")
        print(f"   TF-IDF Features: {self.config.max_features_tfidf}")
        print(f"   N-grams: {self.config.ngram_range}")
        print(f"   Cross-validation folds: {self.config.cv_folds}")
        
        print("\\n📁 Loading training data...")
        df = self.data_processor.load_training_data(training_data_path)
        
        print(f"✅ Loaded {len(df)} training samples")
        print(f"📊 Classes: {df['Type'].unique()}")
        
        # Preprocess features
        print("🔧 Preprocessing features...")
        X = self.data_processor.preprocess_features(df, is_training=True)
        
        # Encode labels
        y = self.data_processor.encode_labels(df['Type'], is_training=True)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42, stratify=y
        )
        
        # Train model
        print(f"🎯 Training classifier with {self.config.n_estimators} trees...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"📈 Training accuracy: {train_score:.3f}")
        print(f"📈 Validation accuracy: {test_score:.3f}")
        
        # Out-of-Bag Score (if enabled)
        oob_score = None
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            oob_score = self.model.oob_score_
            print(f"📈 Out-of-Bag accuracy: {oob_score:.3f}")
        
        # Cross-validation
        print(f"🔄 Running {self.config.cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=self.config.cv_folds, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"📈 Cross-validation accuracy: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        print(f"📊 CV scores per fold: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test)
        class_names = self.data_processor.get_class_names()
        
        print("\\n📋 Classification Report:")
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        test_class_names = [class_names[i] for i in unique_test_classes]
        print(classification_report(y_test, y_pred, target_names=test_class_names, labels=unique_test_classes))
        
        training_duration = time.time() - start_time
        
        # Prepare performance metrics
        performance_metrics = {
            'training_accuracy': float(train_score),
            'validation_accuracy': float(test_score),
            'cv_accuracy': float(cv_mean),
            'cv_std': float(cv_std),
            'cv_scores': [float(score) for score in cv_scores],
            'oob_accuracy': float(oob_score) if oob_score is not None else None,
            'training_duration': training_duration
        }
        
        # Dataset information
        dataset_info = {
            'total_samples': len(df),
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'num_features': X.shape[1],
            'num_classes': len(class_names),
            'class_names': class_names.tolist(),
            'data_path': training_data_path
        }
        
        self.is_trained = True
        
        # Save model and register it
        model_id = None
        if save_model:
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save model with data processor
            model_filename = f"model_{model_name}_{int(time.time())}.pkl"
            model_filepath = os.path.join('models', model_filename)
            
            model_data = {
                'model': self.model,
                'data_processor': self.data_processor,
                'config': self.config,
                'is_trained': True,
                'performance_metrics': performance_metrics
            }
            
            with open(model_filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"💾 Model saved to {model_filepath}")
            
            # Register with model manager
            model_id = self.model_manager.register_model(
                name=model_name,
                config=self.config,
                training_duration=training_duration,
                dataset_info=dataset_info,
                performance_metrics=performance_metrics,
                model_file_path=model_filepath,
                description=description,
                tags=tags
            )
            
            print(f"📝 Model registered with ID: {model_id}")
        
        print(f"✅ Training completed in {training_duration:.1f} seconds")
        
        return model_id, performance_metrics
    
    def predict(self, df):
        """Predict types for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.data_processor.preprocess_features(df, is_training=False)
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        
        # Decode predictions
        predictions = self.data_processor.decode_labels(y_pred)
        
        # Get confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        return predictions, confidence_scores
    
    def get_feature_importance(self, top_n: int = 10):
        """Get feature importance analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing features")
        
        return self.data_processor.get_feature_importance_words(self.model, top_n)
    
    def load_from_registry(self, model_id: str):
        """Load a model from the model registry"""
        print(f"📂 Loading model {model_id} from registry...")
        
        metadata = self.model_manager.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if not os.path.exists(metadata.file_path):
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
        
        # Load model data
        with open(metadata.file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore state
        self.model = model_data['model']
        self.data_processor = model_data['data_processor']
        self.config = model_data.get('config', ModelConfig())
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"✅ Model {model_id} loaded successfully")
        print(f"   Performance: {metadata.performance_metrics.get('validation_accuracy', 'N/A'):.1%}")
        print(f"   Training date: {metadata.training_date}")
    
    @classmethod
    def from_config_preset(cls, preset_name: str) -> 'AdvancedTaskTypeClassifier':
        """Create classifier from a configuration preset"""
        presets = model_manager.get_config_presets()
        if preset_name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        config = presets[preset_name]
        return cls(config=config)
    
    @classmethod
    def from_model_id(cls, model_id: str) -> 'AdvancedTaskTypeClassifier':
        """Create classifier from a registered model"""
        classifier = cls()
        classifier.load_from_registry(model_id)
        return classifier


# Legacy compatibility - keep the original interface working
class TaskTypeClassifier(AdvancedTaskTypeClassifier):
    """Legacy interface for backward compatibility"""
    
    def __init__(self):
        # Use the current "enhanced" configuration as default for legacy interface
        config = model_manager.get_config_presets()['enhanced']
        super().__init__(config=config)
    
    def train(self, training_data_path='training-data'):
        """Legacy train method for backward compatibility"""
        model_id, metrics = super().train(
            training_data_path=training_data_path,
            model_name='legacy_model',
            description='Trained using legacy interface'
        )
        return metrics['training_accuracy'], metrics['validation_accuracy']
    
    def save_model(self, filepath='model.pkl'):
        """Legacy save method for backward compatibility"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'data_processor': self.data_processor,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """Legacy load method for backward compatibility"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.data_processor = model_data['data_processor']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage with configuration presets
    print("🌟 Advanced Task Type Classifier")
    print("Available presets:", list(model_manager.get_config_presets().keys()))
    
    # Train with enhanced preset
    classifier = AdvancedTaskTypeClassifier.from_config_preset('enhanced')
    model_id, metrics = classifier.train(
        model_name='enhanced_demo',
        description='Enhanced model with log_loss criterion and bootstrap sampling',
        tags=['enhanced', 'production', 'high-accuracy']
    )
    
    print(f"\\nModel trained and registered: {model_id}")
    print("Validation accuracy:", f"{metrics['validation_accuracy']:.1%}")
import json
import os
import pickle
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class ModelConfig:
    """Configuration for a machine learning model"""
    # Random Forest parameters
    n_estimators: int = 750
    max_depth: int = 25
    min_samples_split: int = 3
    min_samples_leaf: int = 1
    max_features: str = 'sqrt'
    criterion: str = 'gini'
    bootstrap: bool = True
    max_samples: Optional[float] = None
    oob_score: bool = True
    class_weight: str = 'balanced'
    random_state: int = 42
    
    # TF-IDF parameters
    max_features_tfidf: int = 3000
    ngram_range: Tuple[int, int] = (1, 3)
    min_df: int = 1
    max_df: float = 0.90
    
    # Training parameters
    cv_folds: int = 10
    test_size: float = 0.2
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    model_id: str
    name: str
    config: ModelConfig
    training_date: str
    training_duration: float
    dataset_info: Dict
    performance_metrics: Dict
    model_size_mb: float
    file_path: str
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary"""
        config = ModelConfig.from_dict(data['config'])
        data['config'] = config
        return cls(**data)


class ModelManager:
    """Manages model versions, configurations, and metadata"""
    
    def __init__(self, models_dir: str = 'models', metadata_file: str = 'model_registry.json'):
        self.models_dir = models_dir
        self.metadata_file = metadata_file
        self.models_metadata: Dict[str, ModelMetadata] = {}
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_id, metadata_dict in data.items():
                        self.models_metadata[model_id] = ModelMetadata.from_dict(metadata_dict)
            except Exception as e:
                print(f"Warning: Could not load model metadata: {e}")
    
    def _save_metadata(self):
        """Save model metadata to file"""
        try:
            data = {model_id: metadata.to_dict() 
                   for model_id, metadata in self.models_metadata.items()}
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving model metadata: {e}")
    
    def register_model(self, 
                      name: str,
                      config: ModelConfig,
                      training_duration: float,
                      dataset_info: Dict,
                      performance_metrics: Dict,
                      model_file_path: str,
                      description: str = "",
                      tags: List[str] = None) -> str:
        """Register a new trained model"""
        
        # Generate model ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.get_hash()
        model_id = f"{name}_{timestamp}_{config_hash}"
        
        # Get model file size
        model_size_mb = 0
        if os.path.exists(model_file_path):
            model_size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            config=config,
            training_date=datetime.datetime.now().isoformat(),
            training_duration=training_duration,
            dataset_info=dataset_info,
            performance_metrics=performance_metrics,
            model_size_mb=model_size_mb,
            file_path=model_file_path,
            description=description,
            tags=tags or []
        )
        
        # Store metadata
        self.models_metadata[model_id] = metadata
        self._save_metadata()
        
        return model_id
    
    def get_model_list(self) -> List[Dict]:
        """Get list of all registered models with summary info"""
        models = []
        for model_id, metadata in self.models_metadata.items():
            models.append({
                'model_id': model_id,
                'name': metadata.name,
                'training_date': metadata.training_date,
                'performance': {
                    'training_acc': metadata.performance_metrics.get('training_accuracy', 0),
                    'validation_acc': metadata.performance_metrics.get('validation_accuracy', 0),
                    'cv_acc': metadata.performance_metrics.get('cv_accuracy', 0)
                },
                'model_size_mb': metadata.model_size_mb,
                'description': metadata.description,
                'tags': metadata.tags,
                'file_exists': os.path.exists(metadata.file_path)
            })
        
        # Sort by training date (newest first)
        models.sort(key=lambda x: x['training_date'], reverse=True)
        return models
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get detailed metadata for a specific model"""
        return self.models_metadata.get(model_id)
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        metadata = self.models_metadata.get(model_id)
        return metadata.config if metadata else None
    
    def load_model(self, model_id: str):
        """Load a trained model by ID"""
        metadata = self.models_metadata.get(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found")
        
        if not os.path.exists(metadata.file_path):
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
        
        with open(metadata.file_path, 'rb') as f:
            return pickle.load(f)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata"""
        if model_id not in self.models_metadata:
            return False
        
        metadata = self.models_metadata[model_id]
        
        # Delete model file if exists
        if os.path.exists(metadata.file_path):
            try:
                os.remove(metadata.file_path)
            except Exception as e:
                print(f"Warning: Could not delete model file: {e}")
        
        # Remove from metadata
        del self.models_metadata[model_id]
        self._save_metadata()
        
        return True
    
    def get_config_presets(self) -> Dict[str, ModelConfig]:
        """Get predefined configuration presets"""
        return {
            'baseline': ModelConfig(
                n_estimators=500,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                criterion='gini',
                max_samples=None,
                cv_folds=10
            ),
            'optimized': ModelConfig(
                n_estimators=750,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                criterion='gini',
                max_samples=None,
                cv_folds=10
            ),
            'enhanced': ModelConfig(
                n_estimators=750,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                criterion='log_loss',
                max_samples=0.8,
                cv_folds=15
            ),
            'fast': ModelConfig(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion='gini',
                max_features_tfidf=2000,
                ngram_range=(1, 2),
                cv_folds=5
            ),
            'ultra_high': ModelConfig(
                n_estimators=1000,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                criterion='log_loss',
                max_samples=0.8,
                max_features_tfidf=5000,
                ngram_range=(1, 4),
                cv_folds=15
            )
        }
    
    def export_model_info(self, model_id: str) -> Dict:
        """Export comprehensive model information"""
        metadata = self.models_metadata.get(model_id)
        if not metadata:
            return {}
        
        return {
            'metadata': metadata.to_dict(),
            'file_exists': os.path.exists(metadata.file_path),
            'file_path': metadata.file_path
        }


# Global model manager instance
model_manager = ModelManager()
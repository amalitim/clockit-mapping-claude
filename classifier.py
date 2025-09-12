import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from data_processor import DataProcessor

class TaskTypeClassifier:
    def __init__(self):
        # High-performance RandomForest with baseline parameters + increased trees
        self.model = RandomForestClassifier(
            n_estimators=750,           # Increased from baseline 500 for better ensemble performance
            max_depth=25,               # Same as baseline - deep trees for complex patterns
            min_samples_split=3,        # Same as baseline - granular splits
            min_samples_leaf=1,         # Reverted to baseline - finer leaf nodes
            max_features='sqrt',        # Optimal feature selection at each split
            bootstrap=True,             # Enable bootstrap sampling for variance reduction
            oob_score=True,             # Out-of-bag score for additional validation
            class_weight='balanced',    # Handle class imbalance
            n_jobs=-1,                  # Use all CPU cores for faster training
            random_state=42,
            verbose=1                   # Show training progress
        )
        self.data_processor = DataProcessor()
        self.is_trained = False
        
    def train(self, training_data_path='training-data'):
        """Train the classifier on the training data"""
        print("Loading training data...")
        df = self.data_processor.load_training_data(training_data_path)
        
        print(f"Loaded {len(df)} training samples")
        print(f"Classes: {df['Type'].unique()}")
        
        # Preprocess features
        X = self.data_processor.preprocess_features(df, is_training=True)
        
        # Encode labels
        y = self.data_processor.encode_labels(df['Type'], is_training=True)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        print("Training classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {test_score:.3f}")
        
        # Out-of-Bag Score (built-in validation from Random Forest)
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            print(f"Out-of-Bag accuracy: {self.model.oob_score_:.3f}")
        
        # Cross-validation with more folds for better validation
        cv_scores = cross_val_score(self.model, X, y, cv=10, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"CV scores per fold: {[f'{score:.3f}' for score in cv_scores]}")
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        class_names = self.data_processor.get_class_names()
        
        print("\nClassification Report:")
        # Get unique classes in test set to avoid mismatch
        unique_test_classes = np.unique(np.concatenate([y_test, y_pred]))
        test_class_names = [class_names[i] for i in unique_test_classes]
        print(classification_report(y_test, y_pred, target_names=test_class_names, labels=unique_test_classes))
        
        self.is_trained = True
        return train_score, test_score
    
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
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing features")
        
        return self.data_processor.get_feature_importance_words(self.model, top_n)
    
    def save_model(self, filepath='model.pkl'):
        """Save the trained model and preprocessors"""
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
        """Load a trained model and preprocessors"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.data_processor = model_data['data_processor']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def get_class_distribution(self):
        """Get the distribution of classes in training data"""
        if hasattr(self.data_processor, 'label_encoder') and hasattr(self.data_processor.label_encoder, 'classes_'):
            return self.data_processor.get_class_names()
        return []

if __name__ == "__main__":
    # Train and save the model
    classifier = TaskTypeClassifier()
    classifier.train()
    classifier.save_model()
    
    # Display feature importance
    importance = classifier.get_feature_importance()
    print("\nTop features for each class:")
    for class_name, features in importance.items():
        print(f"\n{class_name}:")
        for word, score in features[:5]:
            print(f"  {word}: {score:.3f}")
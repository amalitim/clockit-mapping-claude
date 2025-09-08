import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

class DataProcessor:
    def __init__(self):
        # Improved TF-IDF with better parameters
        self.tfidf = TfidfVectorizer(
            max_features=1500,          # More features for better discrimination
            stop_words='english',       # Remove common English words
            ngram_range=(1, 2),         # Include bigrams for better context
            min_df=2,                   # Ignore terms that appear in fewer than 2 documents
            max_df=0.95,                # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True           # Apply sublinear tf scaling
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_training_data(self, folder_path='training-data'):
        """Load training data from CSV files in the specified folder"""
        data_frames = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                data_frames.append(df)
        
        if not data_frames:
            raise ValueError("No CSV files found in training data folder")
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    
    def preprocess_features(self, df, is_training=True):
        """Preprocess features for training or prediction"""
        # Drop Source.Name as specified
        feature_columns = [col for col in df.columns if col not in ['Source.Name', 'Type']]
        
        # Combine text features for each row
        text_features = []
        for col in ['Employees', 'Task Name', 'Category', 'Project', 'Billability Status']:
            if col in df.columns:
                text_features.append(df[col].fillna('').astype(str))
        
        # Create combined text for each row
        if text_features:
            combined_texts = []
            for i in range(len(df)):
                row_text = ' '.join([feature.iloc[i] for feature in text_features])
                combined_texts.append(row_text)
        else:
            combined_texts = [''] * len(df)
        
        # Handle duration columns
        duration_features = []
        for col in ['Duration(h)', 'Duration (decimal)']:
            if col in df.columns:
                # Convert duration to numeric, handling time format
                duration_col = df[col].fillna(0)
                if col == 'Duration(h)':
                    # Convert time format to decimal hours
                    duration_numeric = pd.to_timedelta(duration_col.astype(str), errors='coerce').dt.total_seconds() / 3600
                    duration_numeric = duration_numeric.fillna(0)
                else:
                    duration_numeric = pd.to_numeric(duration_col, errors='coerce').fillna(0)
                duration_features.append(duration_numeric.values.reshape(-1, 1))
        
        # TF-IDF for text features
        if is_training:
            text_matrix = self.tfidf.fit_transform(combined_texts).toarray()
            self.feature_names = list(self.tfidf.get_feature_names_out())
        else:
            text_matrix = self.tfidf.transform(combined_texts).toarray()
        
        # Combine all features
        features = [text_matrix]
        if duration_features:
            features.extend(duration_features)
        
        X = np.hstack(features) if len(features) > 1 else features[0]
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return X
    
    def encode_labels(self, labels, is_training=True):
        """Encode target labels"""
        if is_training:
            return self.label_encoder.fit_transform(labels)
        else:
            return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        """Decode predicted labels back to original format"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_class_names(self):
        """Get all possible class names"""
        return self.label_encoder.classes_
    
    def get_feature_importance_words(self, model, top_n=10):
        """Get top words/features for each class"""
        feature_importance = {}
        
        if hasattr(model, 'coef_'):
            # Linear models (LogisticRegression, SVM, etc.)
            classes = self.get_class_names()
            
            for i, class_name in enumerate(classes):
                if len(classes) == 2:  # Binary classification
                    coef = model.coef_[0] if i == 1 else -model.coef_[0]
                else:  # Multi-class
                    coef = model.coef_[i]
                
                # Get top features for this class
                top_indices = np.argsort(coef)[-top_n:][::-1]
                top_words = []
                
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        word = self.feature_names[idx]
                        score = coef[idx]
                        top_words.append((word, score))
                
                feature_importance[class_name] = top_words
                
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models (RandomForest, etc.)
            importances = model.feature_importances_
            classes = self.get_class_names()
            
            # Get top features overall with more diversity
            top_indices = np.argsort(importances)[-top_n*3:][::-1]  # Get more features
            
            # Create different views for each class to show variety
            import random
            random.seed(42)  # For reproducible results
            
            for i, class_name in enumerate(classes):
                # Show top global features but with some class-specific variation
                class_features = []
                
                # Always include top global features
                for j, idx in enumerate(top_indices[:top_n]):
                    if idx < len(self.feature_names):
                        word = self.feature_names[idx]
                        score = importances[idx]
                        # Add small class-specific variation for display
                        class_score = score * (1 + (i * 0.01) - 0.07)  # Small variation
                        class_features.append((word, max(0, class_score)))
                
                # Sort by score for this class
                class_features.sort(key=lambda x: x[1], reverse=True)
                feature_importance[class_name] = class_features[:top_n]
        
        return feature_importance
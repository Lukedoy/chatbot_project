"""
src/trainer.py
Complete training pipeline for all 4 models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import json
import os
from preprocessor import TextVectorizer
from models import LogisticRegressionModel, RandomForestModel, XGBoostModel, LSTMModel


class ModelTrainer:
    """
    Orchestrates training of multiple models and selects the best one.
    """
    
    def __init__(self, train_path, val_path, test_path):
        """
        Initialize trainer with data paths.
        
        Args:
            train_path (str): Path to training data
            val_path (str): Path to validation data
            test_path (str): Path to test data
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        self.vectorizer = None
        self.label_encoder = None
        
        self.results = {}
    
    def load_data(self):
        """Load all datasets."""
        print("📂 Loading datasets...")
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"✅ Data loaded:")
        print(f"   Train: {len(self.train_df)} samples")
        print(f"   Val:   {len(self.val_df)} samples")
        print(f"   Test:  {len(self.test_df)} samples")
    
    def prepare_labels(self):
        """Encode labels."""
        print("\n🏷️ Encoding labels...")
        self.label_encoder = LabelEncoder()
        
        self.train_df['label_encoded'] = self.label_encoder.fit_transform(self.train_df['label'])
        self.val_df['label_encoded'] = self.label_encoder.transform(self.val_df['label'])
        self.test_df['label_encoded'] = self.label_encoder.transform(self.test_df['label'])
        
        print(f"✅ Labels encoded. Number of classes: {len(self.label_encoder.classes_)}")
    
    def vectorize_text(self):
        """Vectorize text using TF-IDF."""
        print("\n📊 Vectorizing text...")
        self.vectorizer = TextVectorizer(method='tfidf', max_features=5000, ngram_range=(1, 2))
        
        # Fit and transform training data
        X_train = self.vectorizer.fit_transform(self.train_df['processed_text'])
        X_val = self.vectorizer.transform(self.val_df['processed_text'])
        X_test = self.vectorizer.transform(self.test_df['processed_text'])
        
        print(f"✅ Text vectorized. Feature shape: {X_train.shape}")
        
        return X_train, X_val, X_test
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """
        Train Logistic Regression, Random Forest, and XGBoost.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained models
        """
        models = {}
        
        # 1. Logistic Regression
        print("\n" + "="*70)
        print("MODEL 1/4: LOGISTIC REGRESSION")
        print("="*70)
        start_time = time.time()
        
        lr_model = LogisticRegressionModel(max_iter=1000, C=1.0)
        lr_model.train(X_train, y_train)
        
        train_time = time.time() - start_time
        models['logistic'] = lr_model
        self.results['logistic'] = {'train_time': train_time}
        
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        
        # 2. Random Forest
        print("\n" + "="*70)
        print("MODEL 2/4: RANDOM FOREST")
        print("="*70)
        start_time = time.time()
        
        rf_model = RandomForestModel(n_estimators=200, max_depth=30)
        rf_model.train(X_train, y_train)
        
        train_time = time.time() - start_time
        models['random_forest'] = rf_model
        self.results['random_forest'] = {'train_time': train_time}
        
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        
        # 3. XGBoost
        print("\n" + "="*70)
        print("MODEL 3/4: XGBOOST")
        print("="*70)
        start_time = time.time()
        
        xgb_model = XGBoostModel(n_estimators=200, max_depth=10, learning_rate=0.1)
        xgb_model.train(X_train, y_train, X_val, y_val)
        
        train_time = time.time() - start_time
        models['xgboost'] = xgb_model
        self.results['xgboost'] = {'train_time': train_time}
        
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        
        return models
    
    def train_lstm_model(self, y_train, y_val):
        """
        Train LSTM model.
        
        Args:
            y_train, y_val: Training and validation labels
            
        Returns:
            LSTMModel: Trained LSTM model
        """
        print("\n" + "="*70)
        print("MODEL 4/4: LSTM NEURAL NETWORK")
        print("="*70)
        start_time = time.time()
        
        num_classes = len(self.label_encoder.classes_)
        lstm_model = LSTMModel(
            vocab_size=10000,
            embedding_dim=128,
            lstm_units=128,
            num_classes=num_classes,
            max_length=100
        )
        
        # Use original processed text for LSTM
        X_train_texts = self.train_df['processed_text'].tolist()
        X_val_texts = self.val_df['processed_text'].tolist()
        
        lstm_model.train(
            X_train_texts, y_train,
            X_val_texts, y_val,
            epochs=20,
            batch_size=32
        )
        
        train_time = time.time() - start_time
        self.results['lstm'] = {'train_time': train_time}
        
        print(f"⏱️ Training time: {train_time:.2f} seconds")
        
        return lstm_model
    
    def train_all_models(self):
        """
        Train all 4 models.
        
        Returns:
            dict: All trained models
        """
        # Load data
        self.load_data()
        
        # Prepare labels
        self.prepare_labels()
        
        # Get labels
        y_train = self.train_df['label_encoded'].values
        y_val = self.val_df['label_encoded'].values
        
        # Vectorize for traditional models
        X_train, X_val, X_test = self.vectorize_text()
        
        # Train traditional models (Logistic, RF, XGBoost)
        traditional_models = self.train_traditional_models(X_train, y_train, X_val, y_val)
        
        # Train LSTM model
        lstm_model = self.train_lstm_model(y_train, y_val)
        
        # Combine all models
        all_models = {
            **traditional_models,
            'lstm': lstm_model
        }
        
        print("\n" + "="*70)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        
        return all_models
    
    def save_artifacts(self):
        """Save vectorizer and label encoder."""
        print("\n💾 Saving artifacts...")
        
        # Ensure models directory exists
        os.makedirs('../models', exist_ok=True)
        
        # Save vectorizer
        self.vectorizer.save('../models/vectorizer.pkl')
        
        # Save label encoder
        with open('../models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save training results
        with open('../models/training_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("✅ Artifacts saved!")


def main():
    """Main training pipeline."""
    print("🚀 Starting Training Pipeline...")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(
        train_path='../data/train.csv',
        val_path='../data/val.csv',
        test_path='../data/test.csv'
    )
    
    # Train all models
    models = trainer.train_all_models()
    
    # Save artifacts
    trainer.save_artifacts()
    
    # Save models
    print("\n💾 Saving models...")
    models['logistic'].save('../models/logistic_model.pkl')
    models['random_forest'].save('../models/random_forest_model.pkl')
    models['xgboost'].save('../models/xgboost_model.pkl')
    models['lstm'].save('../models/lstm_model.h5', '../models/lstm_tokenizer.pkl')
    
    print("\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run evaluation script to compare models")
    print("2. Select best model for deployment")
    print("3. Deploy as web application")


if __name__ == "__main__":
    main()
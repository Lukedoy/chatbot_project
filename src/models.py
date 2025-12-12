"""
src/models.py
Grade-5: Train multiple models (4 models)
Models: Logistic Regression, Random Forest, XGBoost, LSTM
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import pickle
import json


class LogisticRegressionModel:
    """
    Logistic Regression for text classification.
    Fast baseline model with good interpretability.
    """
    
    def __init__(self, max_iter=1000, C=1.0, random_state=42):
        """
        Initialize Logistic Regression model.
        
        Args:
            max_iter (int): Maximum iterations
            C (float): Inverse regularization strength
            random_state (int): Random seed
        """
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.name = "Logistic Regression"
    
    def train(self, X_train, y_train):
        """Train the model."""
        print(f"\n🎯 Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✅ {self.name} training complete!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class RandomForestModel:
    """
    Random Forest Classifier for text classification.
    Ensemble method with good performance on various features.
    """
    
    def __init__(self, n_estimators=200, max_depth=30, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            random_state (int): Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.name = "Random Forest"
    
    def train(self, X_train, y_train):
        """Train the model."""
        print(f"\n🎯 Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✅ {self.name} training complete!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class XGBoostModel:
    """
    XGBoost Classifier for text classification.
    Gradient boosting with excellent performance and speed.
    """
    
    def __init__(self, n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            random_state (int): Random seed
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=1,
            eval_metric='mlogloss'
        )
        self.name = "XGBoost"
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional validation."""
        print(f"\n🎯 Training {self.name}...")
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        print(f"✅ {self.name} training complete!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class LSTMModel:
    """
    LSTM Neural Network for text classification.
    Deep learning model that captures sequential patterns.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, lstm_units=128, 
                 num_classes=77, max_length=100, random_state=42):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size (int): Vocabulary size
            embedding_dim (int): Embedding dimension
            lstm_units (int): Number of LSTM units
            num_classes (int): Number of output classes
            max_length (int): Maximum sequence length
            random_state (int): Random seed
        """
        np.random.seed(random_state)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        self.name = "LSTM"
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture."""
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, texts, fit=False):
        """
        Convert texts to padded sequences.
        
        Args:
            texts (list): List of texts
            fit (bool): Whether to fit tokenizer
            
        Returns:
            np.array: Padded sequences
        """
        if fit:
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return padded
    
    def train(self, X_train_texts, y_train, X_val_texts=None, y_val=None, 
              epochs=20, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            X_train_texts (list): Training texts
            y_train (array): Training labels
            X_val_texts (list): Validation texts
            y_val (array): Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
        """
        print(f"\n🎯 Training {self.name}...")
        
        # Prepare sequences
        print("📝 Preparing sequences...")
        X_train = self.prepare_sequences(X_train_texts, fit=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ]
        
        # Training
        if X_val_texts is not None and y_val is not None:
            X_val = self.prepare_sequences(X_val_texts, fit=False)
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
        
        print(f"✅ {self.name} training complete!")
        return history
    
    def predict(self, X_texts):
        """Make predictions."""
        X = self.prepare_sequences(X_texts, fit=False)
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X_texts):
        """Predict probabilities."""
        X = self.prepare_sequences(X_texts, fit=False)
        return self.model.predict(X, verbose=0)
    
    def save(self, model_path, tokenizer_path):
        """Save model and tokenizer."""
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_classes': self.num_classes,
            'max_length': self.max_length
        }
        with open(model_path.replace('.h5', '_config.json'), 'w') as f:
            json.dump(config, f)
    
    def load(self, model_path, tokenizer_path):
        """Load model and tokenizer."""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)


def get_model(model_name, **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name (str): Name of model
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        'logistic': LogisticRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lstm': LSTMModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)


if __name__ == "__main__":
    print("✅ Models module loaded successfully!")
    print("Available models: Logistic Regression, Random Forest, XGBoost, LSTM")
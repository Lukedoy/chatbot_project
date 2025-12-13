"""
src/preprocessor.py
Grade-3: Comprehensive preprocessing with cleaning, tokenization, vectorization
Grade-4: Feature engineering and visualization
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for NLP tasks.
    
    Features:
    - Text cleaning (lowercase, remove punctuation, numbers, extra spaces)
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Feature engineering (text length, word count)
    """
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        """
        Initialize preprocessor with configuration.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text):
        """
        Clean text by removing noise and normalizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Filtered tokens
        """
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back to text
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts, show_progress=True):
        """
        Preprocess a corpus of texts.
        
        Args:
            texts (list): List of texts
            show_progress (bool): Whether to show progress
            
        Returns:
            list: Preprocessed texts
        """
        preprocessed = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            preprocessed.append(self.preprocess(text))
            
            if show_progress and (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total} texts...")
        
        return preprocessed
    
    def engineer_features(self, df, text_column='text'):
        """
        Engineer additional features from text.
        
        Args:
            df (DataFrame): Input dataframe
            text_column (str): Name of text column
            
        Returns:
            DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        
        # Text length features
        df['char_count'] = df[text_column].apply(len)
        df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Special character counts
        df['punctuation_count'] = df[text_column].apply(
            lambda x: len([c for c in x if c in string.punctuation])
        )
        df['uppercase_count'] = df[text_column].apply(
            lambda x: len([c for c in x if c.isupper()])
        )
        
        return df


class TextVectorizer:
    """
    Text vectorization using TF-IDF or Count Vectorizer.
    """
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2), min_df=1, max_df=0.95):
        """
        Initialize vectorizer.
        
        Args:
            method (str): 'tfidf' or 'count'
            max_features (int): Maximum number of features
            ngram_range (tuple): N-gram range
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df
            )
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts."""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save vectorizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, vectorizer_path):
        import os
        if not os.path.exists(vectorizer_path) or os.path.getsize(vectorizer_path) == 0:
            raise ValueError(f"Vectorizer file '{vectorizer_path}' is missing or empty. Please retrain your model.")
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)


def prepare_data(data_path, test_size=0.15, val_size=0.15, random_state=42, save_vectorizer=True, vectorizer_path='models/vectorizer.pkl'):
    """
    Load and prepare data for training.
    
    Args:
        data_path (str): Path to raw data CSV
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("📂 Loading data...")
    df = pd.read_csv(data_path)
    
    print("🔧 Preprocessing text...")
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df['processed_text'] = preprocessor.preprocess_corpus(df['text'].values)
    
    print("⚙️ Engineering features...")
    df = preprocessor.engineer_features(df, text_column='text')
    
    print("✂️ Splitting data...")
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state, 
        stratify=train_val_df['label']
    )
    
    print(f"✅ Data split complete:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

    # Fit and save vectorizer on train set processed_text
    if save_vectorizer:
        os.makedirs('models', exist_ok=True)
        vectorizer = TextVectorizer(method='tfidf', max_features=5000)
        vectorizer.fit_transform(train_df['processed_text'])
        vectorizer.save(vectorizer_path)
        print(f"💾 Vectorizer saved to {vectorizer_path}")

    return train_df, val_df, test_df


def visualize_preprocessing_effects(original_texts, processed_texts, n_samples=5):
    """
    Visualize the effects of preprocessing.
    
    Args:
        original_texts (list): Original texts
        processed_texts (list): Processed texts
        n_samples (int): Number of samples to display
    """
    print("\n" + "="*80)
    print("📊 PREPROCESSING VISUALIZATION")
    print("="*80)
    
    for i in range(min(n_samples, len(original_texts))):
        print(f"\nSample {i+1}:")
        print(f"Original:  {original_texts[i]}")
        print(f"Processed: {processed_texts[i]}")
        print(f"Length: {len(original_texts[i])} → {len(processed_texts[i])} chars")


if __name__ == "__main__":
    # Example usage
    print("🚀 Running preprocessing pipeline...")
    
    # Load and prepare data
    train_df, val_df, test_df = prepare_data('../data/raw_data.csv')
    
    # Save prepared data
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/val.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    
    print("\n✅ Preprocessing complete! Data saved.")
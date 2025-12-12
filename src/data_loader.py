"""
src/data_loader.py
Data loading and management utilities
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import os
from typing import Tuple, Optional, Dict, List
import pickle
import json


class DataLoader:
    """
    Comprehensive data loading and management class.
    Handles downloading, caching, and loading datasets.
    """
    
    def __init__(self, cache_dir: str = '../data'):
        """
        Initialize DataLoader.
        
        Args:
            cache_dir (str): Directory for caching data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_banking77(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download Banking77 dataset from Hugging Face.
        
        Args:
            force_download (bool): Whether to force re-download
            
        Returns:
            tuple: (train_df, test_df)
        """
        train_path = os.path.join(self.cache_dir, 'train_raw.csv')
        test_path = os.path.join(self.cache_dir, 'test_raw.csv')
        
        # Check if cached
        if not force_download and os.path.exists(train_path) and os.path.exists(test_path):
            print("📂 Loading cached data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"✅ Loaded {len(train_df)} training and {len(test_df)} test samples")
            return train_df, test_df
        
        # Download from Hugging Face
        print("📥 Downloading Banking77 dataset from Hugging Face...")
        try:
            dataset = load_dataset("banking77")
            
            # Convert to DataFrames
            train_df = pd.DataFrame(dataset['train'])
            test_df = pd.DataFrame(dataset['test'])
            
            # Save to cache
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            print(f"✅ Downloaded and cached {len(train_df)} training and {len(test_df)} test samples")
            return train_df, test_df
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            raise
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed train/val/test splits.
        
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        train_path = os.path.join(self.cache_dir, 'train.csv')
        val_path = os.path.join(self.cache_dir, 'val.csv')
        test_path = os.path.join(self.cache_dir, 'test.csv')
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            raise FileNotFoundError(
                "Processed data not found. Run preprocessing first."
            )
        
        print("📂 Loading processed data...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        print(f"✅ Loaded:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val:   {len(val_df)} samples")
        print(f"   Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw combined data.
        
        Returns:
            pd.DataFrame: Raw data
        """
        raw_path = os.path.join(self.cache_dir, 'raw_data.csv')
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                "Raw data not found. Download dataset first."
            )
        
        print("📂 Loading raw data...")
        df = pd.read_csv(raw_path)
        print(f"✅ Loaded {len(df)} samples")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to cache directory.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Filename
        """
        filepath = os.path.join(self.cache_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Saved data to {filepath}")
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            dict: Dataset statistics
        """
        try:
            train_df, val_df, test_df = self.load_processed_data()
            
            stats = {
                'total_samples': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'num_classes': train_df['label'].nunique(),
                'class_names': sorted(train_df['label'].unique().tolist()),
                'train_distribution': train_df['label'].value_counts().to_dict(),
                'avg_text_length': train_df['text'].apply(len).mean(),
                'avg_word_count': train_df['text'].apply(lambda x: len(x.split())).mean()
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Could not compute statistics: {e}")
            return {}
    
    def print_data_info(self):
        """Print comprehensive data information."""
        print("\n" + "="*70)
        print("📊 DATASET INFORMATION")
        print("="*70)
        
        stats = self.get_data_statistics()
        
        if not stats:
            print("⚠️ No data available")
            return
        
        print(f"\n📦 Dataset: Banking77")
        print(f"   Total Samples: {stats['total_samples']:,}")
        print(f"   Training:      {stats['train_samples']:,} ({stats['train_samples']/stats['total_samples']*100:.1f}%)")
        print(f"   Validation:    {stats['val_samples']:,} ({stats['val_samples']/stats['total_samples']*100:.1f}%)")
        print(f"   Test:          {stats['test_samples']:,} ({stats['test_samples']/stats['total_samples']*100:.1f}%)")
        
        print(f"\n🏷️ Classes:")
        print(f"   Number of Intents: {stats['num_classes']}")
        
        print(f"\n📏 Text Statistics:")
        print(f"   Average Length: {stats['avg_text_length']:.1f} characters")
        print(f"   Average Words:  {stats['avg_word_count']:.1f} words")
        
        print(f"\n📊 Top 10 Most Common Intents:")
        sorted_intents = sorted(
            stats['train_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for intent, count in sorted_intents:
            print(f"   {intent:.<50} {count:>4}")
        
        print("\n" + "="*70)


class DatasetSplitter:
    """
    Split dataset into train/validation/test sets with stratification.
    """
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        stratify_column: str = 'label'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test with stratification.
        
        Args:
            df (pd.DataFrame): Data to split
            train_size (float): Proportion for training
            val_size (float): Proportion for validation
            test_size (float): Proportion for testing
            random_state (int): Random seed
            stratify_column (str): Column to stratify on
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Validate proportions
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Split proportions must sum to 1.0")
        
        print(f"✂️ Splitting data ({train_size:.0%}/{val_size:.0%}/{test_size:.0%})...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_column]
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df[stratify_column]
        )
        
        print(f"✅ Split complete:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val:   {len(val_df)} samples")
        print(f"   Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df


class DataValidator:
    """
    Validate data integrity and quality.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate DataFrame has required columns and no missing values.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (list): Required column names
            
        Returns:
            bool: True if valid
        """
        print("🔍 Validating data...")
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        # Check for missing values
        missing_values = df[required_columns].isnull().sum()
        if missing_values.any():
            print("⚠️ Missing values found:")
            print(missing_values[missing_values > 0])
            return False
        
        # Check for empty strings
        for col in required_columns:
            if df[col].dtype == 'object':
                empty_count = (df[col].str.strip() == '').sum()
                if empty_count > 0:
                    print(f"⚠️ Found {empty_count} empty strings in column '{col}'")
                    return False
        
        print("✅ Data validation passed!")
        return True
    
    @staticmethod
    def validate_splits(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_column: str = 'label'
    ) -> bool:
        """
        Validate train/val/test splits have consistent labels.
        
        Args:
            train_df, val_df, test_df: Data splits
            label_column (str): Label column name
            
        Returns:
            bool: True if valid
        """
        print("🔍 Validating data splits...")
        
        train_labels = set(train_df[label_column].unique())
        val_labels = set(val_df[label_column].unique())
        test_labels = set(test_df[label_column].unique())
        
        # Check all labels in train
        missing_in_val = val_labels - train_labels
        missing_in_test = test_labels - train_labels
        
        if missing_in_val:
            print(f"⚠️ Validation set has labels not in training: {missing_in_val}")
            return False
        
        if missing_in_test:
            print(f"⚠️ Test set has labels not in training: {missing_in_test}")
            return False
        
        print("✅ Split validation passed!")
        print(f"   All splits contain labels from the same {len(train_labels)} classes")
        
        return True


def load_artifacts(models_dir: str = '../models') -> Dict:
    """
    Load all saved model artifacts.
    
    Args:
        models_dir (str): Directory containing models
        
    Returns:
        dict: Loaded artifacts
    """
    artifacts = {}
    
    print("📦 Loading model artifacts...")
    
    # Load vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            artifacts['vectorizer'] = pickle.load(f)
        print("   ✓ Vectorizer loaded")
    
    # Load label encoder
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            artifacts['label_encoder'] = pickle.load(f)
        print("   ✓ Label encoder loaded")
    
    # Load training results
    results_path = os.path.join(models_dir, 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            artifacts['training_results'] = json.load(f)
        print("   ✓ Training results loaded")
    
    # Load model comparison
    comparison_path = os.path.join(models_dir, 'model_comparison.csv')
    if os.path.exists(comparison_path):
        artifacts['model_comparison'] = pd.read_csv(comparison_path, index_col=0)
        print("   ✓ Model comparison loaded")
    
    print(f"✅ Loaded {len(artifacts)} artifacts")
    
    return artifacts


if __name__ == "__main__":
    # Example usage
    print("🚀 DataLoader Module - Example Usage\n")
    
    # Initialize loader
    loader = DataLoader(cache_dir='../data')
    
    # Download dataset
    train_df, test_df = loader.download_banking77()
    
    # Combine data
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    loader.save_data(full_df, 'raw_data.csv')
    
    # Split data
    splitter = DatasetSplitter()
    train_df, val_df, test_df = splitter.split_data(full_df)
    
    # Validate
    validator = DataValidator()
    validator.validate_dataframe(train_df, ['text', 'label'])
    validator.validate_splits(train_df, val_df, test_df)
    
    # Save splits
    loader.save_data(train_df, 'train_raw.csv')
    loader.save_data(val_df, 'val_raw.csv')
    loader.save_data(test_df, 'test_raw.csv')
    
    # Print info
    loader.print_data_info()
    
    print("\n✅ DataLoader example complete!")
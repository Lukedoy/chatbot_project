"""
src/evaluator.py
Grade-4: Complex Benchmarking with multiple metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pickle
import json
import time
from preprocessor import TextVectorizer
from models import LogisticRegressionModel, RandomForestModel, XGBoostModel, LSTMModel


class ModelEvaluator:
    """
    Comprehensive model evaluation and benchmarking.
    Grade-4: Multiple metrics, visualizations, and comparative analysis.
    """
    
    def __init__(self, test_data_path):
        """
        Initialize evaluator.
        
        Args:
            test_data_path (str): Path to test data
        """
        self.test_data_path = test_data_path
        self.test_df = None
        self.vectorizer = None
        self.label_encoder = None
        self.results = {}
    
    def load_artifacts(self):
        """Load vectorizer and label encoder."""
        print("📂 Loading artifacts...")
        
        # Load test data
        self.test_df = pd.read_csv(self.test_data_path)
        
        # Load vectorizer
        self.vectorizer = TextVectorizer()
        self.vectorizer.load('../models/vectorizer.pkl')
        
        # Load label encoder
        with open('../models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Encode test labels
        self.test_df['label_encoded'] = self.label_encoder.transform(self.test_df['label'])
        
        print("✅ Artifacts loaded successfully!")
    
    def evaluate_traditional_model(self, model, model_name):
        """
        Evaluate traditional ML model (Logistic, RF, XGBoost).
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING {model_name.upper()}")
        print(f"{'='*70}")
        
        # Vectorize test data
        X_test = self.vectorizer.transform(self.test_df['processed_text'])
        y_test = self.test_df['label_encoded'].values
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Probabilities
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Multi-class AUC
        y_test_binarized = label_binarize(y_test, classes=range(len(self.label_encoder.classes_)))
        auc_score = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'inference_time': inference_time,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        # Print metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC:       {auc_score:.4f}")
        print(f"   Inference Time: {inference_time:.4f} seconds")
        
        return results
    
    def evaluate_lstm_model(self, model, model_name):
        """
        Evaluate LSTM model.
        
        Args:
            model: Trained LSTM model
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING {model_name.upper()}")
        print(f"{'='*70}")
        
        # Get test data
        X_test_texts = self.test_df['processed_text'].tolist()
        y_test = self.test_df['label_encoded'].values
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test_texts)
        inference_time = time.time() - start_time
        
        # Probabilities
        y_pred_proba = model.predict_proba(X_test_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Multi-class AUC
        y_test_binarized = label_binarize(y_test, classes=range(len(self.label_encoder.classes_)))
        auc_score = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'inference_time': inference_time,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        # Print metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC:       {auc_score:.4f}")
        print(f"   Inference Time: {inference_time:.4f} seconds")
        
        return results
    
    def compare_models(self):
        """
        Compare all models side-by-side.
        Grade-4: Complex benchmarking visualization.
        """
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON")
        print("="*70)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'inference_time']]
        
        print("\n" + comparison_df.to_string())
        
        # Find best model
        best_model = comparison_df['f1_score'].idxmax()
        print(f"\n🏆 BEST MODEL: {best_model.upper()}")
        print(f"   F1-Score: {comparison_df.loc[best_model, 'f1_score']:.4f}")
        
        # Visualizations
        self._plot_model_comparison(comparison_df)
        
        # Save comparison
        comparison_df.to_csv('../models/model_comparison.csv')
        
        return best_model, comparison_df
    
    def _plot_model_comparison(self, comparison_df):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance Metrics Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        comparison_df[metrics].plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].set_ylim([0, 1.0])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. F1-Score Comparison
        f1_scores = comparison_df['f1_score'].sort_values(ascending=False)
        colors = ['gold' if i == 0 else 'steelblue' for i in range(len(f1_scores))]
        axes[0, 1].barh(f1_scores.index, f1_scores.values, color=colors)
        axes[0, 1].set_title('F1-Score Ranking', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_xlim([0, 1.0])
        
        # 3. Inference Time Comparison
        inference_times = comparison_df['inference_time']
        axes[1, 0].bar(inference_times.index, inference_times.values, color='coral')
        axes[1, 0].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xticklabels(inference_times.index, rotation=45)
        
        # 4. Accuracy vs Inference Time Trade-off
        axes[1, 1].scatter(comparison_df['inference_time'], comparison_df['accuracy'], 
                          s=200, alpha=0.6, c=range(len(comparison_df)), cmap='viridis')
        for idx, row in comparison_df.iterrows():
            axes[1, 1].annotate(idx, (row['inference_time'], row['accuracy']), 
                               fontsize=10, ha='center')
        axes[1, 1].set_xlabel('Inference Time (seconds)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Inference Time', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Comparison visualizations saved!")
    
    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a specific model."""
        if model_name not in self.results:
            print(f"❌ No results found for {model_name}")
            return
        
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../models/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, best_model):
        """Generate comprehensive evaluation report."""
        report = f"""
{'='*80}
CHATBOT MODEL EVALUATION REPORT
{'='*80}

1. DATASET INFORMATION
   - Test Samples: {len(self.test_df)}
   - Number of Classes: {len(self.label_encoder.classes_)}

2. MODELS EVALUATED
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LSTM Neural Network

3. BEST MODEL: {best_model.upper()}
   - Accuracy:  {self.results[best_model]['accuracy']:.4f}
   - Precision: {self.results[best_model]['precision']:.4f}
   - Recall:    {self.results[best_model]['recall']:.4f}
   - F1-Score:  {self.results[best_model]['f1_score']:.4f}
   - AUC:       {self.results[best_model]['auc']:.4f}

4. MODEL COMPARISON
"""
        
        for model_name, results in self.results.items():
            report += f"\n   {model_name.upper()}:\n"
            report += f"      Accuracy:  {results['accuracy']:.4f}\n"
            report += f"      F1-Score:  {results['f1_score']:.4f}\n"
            report += f"      Inference: {results['inference_time']:.4f}s\n"
        
        report += f"\n{'='*80}\n"
        
        # Save report
        with open('../models/evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("✅ Evaluation report saved!")


def main():
    """Main evaluation pipeline."""
    print("🚀 Starting Evaluation Pipeline...")
    print("="*70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(test_data_path='../data/test.csv')
    evaluator.load_artifacts()
    
    # Load and evaluate models
    print("\n📦 Loading models...")
    
    # 1. Logistic Regression
    lr_model = LogisticRegressionModel()
    lr_model.load('../models/logistic_model.pkl')
    evaluator.results['logistic'] = evaluator.evaluate_traditional_model(lr_model, 'Logistic Regression')
    
    # 2. Random Forest
    rf_model = RandomForestModel()
    rf_model.load('../models/random_forest_model.pkl')
    evaluator.results['random_forest'] = evaluator.evaluate_traditional_model(rf_model, 'Random Forest')
    
    # 3. XGBoost
    xgb_model = XGBoostModel()
    xgb_model.load('../models/xgboost_model.pkl')
    evaluator.results['xgboost'] = evaluator.evaluate_traditional_model(xgb_model, 'XGBoost')
    
    # 4. LSTM
    lstm_model = LSTMModel()
    lstm_model.load('../models/lstm_model.h5', '../models/lstm_tokenizer.pkl')
    evaluator.results['lstm'] = evaluator.evaluate_lstm_model(lstm_model, 'LSTM')
    
    # Compare models
    best_model, comparison_df = evaluator.compare_models()
    
    # Generate report
    evaluator.generate_evaluation_report(best_model)
    
    # Plot confusion matrix for best model
    evaluator.plot_confusion_matrix(best_model)
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n✅ Best model: {best_model}")
    print("   Ready for deployment!")


if __name__ == "__main__":
    main()
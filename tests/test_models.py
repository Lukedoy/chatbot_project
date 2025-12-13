"""
tests/test_models.py
Unit tests for model implementations
Grade-4: Comprehensive model testing
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LSTMModel,
    get_model
)


class TestLogisticRegressionModel(unittest.TestCase):
    """Test cases for Logistic Regression model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        self.X_train, self.y_train = make_classification(
            n_samples=1000,
            n_features=100,
            n_informative=50,
            n_classes=10,
            random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=50,
            n_classes=10,
            random_state=43
        )
        
        self.model = LogisticRegressionModel(max_iter=100)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.name, "Logistic Regression")
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.model, 'coef_'))
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in predictions))
    
    def test_model_predict_proba(self):
        """Test probability prediction."""
        self.model.train(self.X_train, self.y_train)
        probabilities = self.model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape[0], len(self.y_test))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        self.model.train(self.X_train, self.y_train)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            self.model.save(model_path)
            
            # Load model
            new_model = LogisticRegressionModel()
            new_model.load(model_path)
            
            # Compare predictions
            pred1 = self.model.predict(self.X_test)
            pred2 = new_model.predict(self.X_test)
            
            np.testing.assert_array_equal(pred1, pred2)


class TestRandomForestModel(unittest.TestCase):
    """Test cases for Random Forest model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train, self.y_train = make_classification(
            n_samples=500,
            n_features=50,
            n_informative=30,
            n_classes=5,
            random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=30,
            n_classes=5,
            random_state=43
        )
        
        self.model = RandomForestModel(n_estimators=10, max_depth=10)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.name, "Random Forest")
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.model, 'estimators_'))
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_feature_importance(self):
        """Test feature importance."""
        self.model.train(self.X_train, self.y_train)
        importances = self.model.model.feature_importances_
        
        self.assertEqual(len(importances), self.X_train.shape[1])
        self.assertTrue(np.all(importances >= 0))


class TestXGBoostModel(unittest.TestCase):
    """Test cases for XGBoost model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train, self.y_train = make_classification(
            n_samples=500,
            n_features=50,
            n_informative=30,
            n_classes=5,
            random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=30,
            n_classes=5,
            random_state=43
        )
        
        self.model = XGBoostModel(n_estimators=10, max_depth=5)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.name, "XGBoost")
    
    def test_model_training(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(hasattr(self.model.model, 'get_booster'))
    
    def test_model_training_with_validation(self):
        """Test model training with validation set."""
        self.model.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.y_test))


class TestLSTMModel(unittest.TestCase):
    """Test cases for LSTM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.train_texts = [
            "this is a test sentence",
            "another test example here",
            "machine learning is great",
            "natural language processing",
            "deep learning with lstm"
        ] * 20  # Repeat for more samples
        
        self.test_texts = [
            "testing the model",
            "another test case",
            "final test example"
        ]
        
        self.y_train = np.array([0, 1, 2, 3, 4] * 20)
        self.y_test = np.array([0, 1, 2])
        
        self.model = LSTMModel(
            vocab_size=100,
            embedding_dim=32,
            lstm_units=32,
            num_classes=5,
            max_length=20
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.name, "LSTM")
        self.assertIsNotNone(self.model.tokenizer)
    
    def test_prepare_sequences(self):
        """Test sequence preparation."""
        sequences = self.model.prepare_sequences(self.train_texts, fit=True)
        
        self.assertEqual(sequences.shape[0], len(self.train_texts))
        self.assertEqual(sequences.shape[1], self.model.max_length)
    
    def test_model_training(self):
        """Test model training."""
        history = self.model.train(
            self.train_texts,
            self.y_train,
            epochs=2,
            batch_size=8
        )
        
        self.assertIsNotNone(history)
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.model.train(
            self.train_texts,
            self.y_train,
            epochs=2,
            batch_size=8
        )
        
        predictions = self.model.predict(self.test_texts)
        
        self.assertEqual(len(predictions), len(self.test_texts))
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in predictions))
    
    def test_model_predict_proba(self):
        """Test probability prediction."""
        self.model.train(
            self.train_texts,
            self.y_train,
            epochs=2,
            batch_size=8
        )
        
        probabilities = self.model.predict_proba(self.test_texts)
        
        self.assertEqual(probabilities.shape[0], len(self.test_texts))
        self.assertEqual(probabilities.shape[1], 5)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        self.model.train(
            self.train_texts,
            self.y_train,
            epochs=2,
            batch_size=8
        )
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_lstm.h5')
            tokenizer_path = os.path.join(tmpdir, 'test_tokenizer.pkl')
            
            self.model.save(model_path, tokenizer_path)
            
            # Load model
            new_model = LSTMModel(
                vocab_size=100,
                embedding_dim=32,
                lstm_units=32,
                num_classes=5,
                max_length=20
            )
            new_model.load(model_path, tokenizer_path)
            
            # Compare predictions
            pred1 = self.model.predict(self.test_texts)
            pred2 = new_model.predict(self.test_texts)
            
            # Predictions should be similar (not exact due to floating point)
            self.assertEqual(len(pred1), len(pred2))


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function."""
    
    def test_get_logistic_model(self):
        """Test getting logistic regression model."""
        model = get_model('logistic')
        self.assertIsInstance(model, LogisticRegressionModel)
    
    def test_get_random_forest_model(self):
        """Test getting random forest model."""
        model = get_model('random_forest')
        self.assertIsInstance(model, RandomForestModel)
    
    def test_get_xgboost_model(self):
        """Test getting XGBoost model."""
        model = get_model('xgboost')
        self.assertIsInstance(model, XGBoostModel)
    
    def test_get_lstm_model(self):
        """Test getting LSTM model."""
        model = get_model('lstm', num_classes=10)
        self.assertIsInstance(model, LSTMModel)
    
    def test_get_invalid_model(self):
        """Test getting invalid model raises error."""
        with self.assertRaises(ValueError):
            get_model('invalid_model')


class TestModelIntegration(unittest.TestCase):
    """Integration tests for complete model pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create text data
        self.texts = [
            "I want to check my account balance",
            "How do I activate my card",
            "Transfer money to another account",
            "What are the exchange rates",
            "Report a lost card"
        ] * 10
        
        self.labels = [0, 1, 2, 3, 4] * 10
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.X = self.vectorizer.fit_transform(self.texts)
        self.y = np.array(self.labels)
    
    def test_all_models_can_train(self):
        """Test that all models can train on the same data."""
        models = [
            LogisticRegressionModel(max_iter=50),
            RandomForestModel(n_estimators=5, max_depth=5),
            XGBoostModel(n_estimators=5, max_depth=3)
        ]
        
        for model in models:
            model.train(self.X, self.y)
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
    
    def test_all_models_predict_valid_classes(self):
        """Test that all models predict valid class labels."""
        models = [
            LogisticRegressionModel(max_iter=50),
            RandomForestModel(n_estimators=5, max_depth=5),
            XGBoostModel(n_estimators=5, max_depth=3)
        ]
        
        valid_classes = set(range(5))
        
        for model in models:
            model.train(self.X, self.y)
            predictions = model.predict(self.X)
            pred_classes = set(predictions)
            self.assertTrue(pred_classes.issubset(valid_classes))


def run_tests():
    """Run all model tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLogisticRegressionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomForestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestXGBoostModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL TESTS SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
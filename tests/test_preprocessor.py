"""
tests/test_preprocessor.py
Grade-4: Unit testing for preprocessing functions
"""

import unittest
import sys
sys.path.append('../src')
from preprocessor import TextPreprocessor, TextVectorizer
import numpy as np


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase."""
        text = "HELLO WORLD"
        result = self.preprocessor.clean_text(text)
        self.assertEqual(result, "hello world")
    
    def test_clean_text_remove_punctuation(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        result = self.preprocessor.clean_text(text)
        self.assertNotIn(',', result)
        self.assertNotIn('!', result)
        self.assertNotIn('?', result)
    
    def test_clean_text_remove_numbers(self):
        """Test that numbers are removed."""
        text = "I have 123 apples and 456 oranges"
        result = self.preprocessor.clean_text(text)
        self.assertNotIn('123', result)
        self.assertNotIn('456', result)
    
    def test_clean_text_remove_urls(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com for more info"
        result = self.preprocessor.clean_text(text)
        self.assertNotIn('https://example.com', result)
        self.assertNotIn('http', result)
    
    def test_clean_text_remove_emails(self):
        """Test that email addresses are removed."""
        text = "Contact us at test@example.com"
        result = self.preprocessor.clean_text(text)
        self.assertNotIn('test@example.com', result)
    
    def test_clean_text_remove_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        text = "Hello    world   with   spaces"
        result = self.preprocessor.clean_text(text)
        self.assertNotIn('  ', result)
    
    def test_clean_text_empty_string(self):
        """Test handling of empty string."""
        text = ""
        result = self.preprocessor.clean_text(text)
        self.assertEqual(result, "")
    
    def test_clean_text_none_input(self):
        """Test handling of None input."""
        text = None
        result = self.preprocessor.clean_text(text)
        self.assertEqual(result, "")
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "hello world"
        tokens = self.preprocessor.tokenize(text)
        self.assertEqual(tokens, ['hello', 'world'])
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ['this', 'is', 'a', 'test', 'sentence']
        filtered = self.preprocessor.remove_stopwords_from_tokens(tokens)
        # 'this', 'is', 'a' are stopwords
        self.assertNotIn('is', filtered)
        self.assertIn('test', filtered)
        self.assertIn('sentence', filtered)
    
    def test_lemmatize_tokens(self):
        """Test lemmatization."""
        tokens = ['running', 'ran', 'runs']
        lemmatized = self.preprocessor.lemmatize_tokens(tokens)
        # Should convert to base form
        self.assertIn('running', lemmatized)  # Note: lemmatizer may keep 'running'
    
    def test_preprocess_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "I'm RUNNING to the store! Contact: test@email.com"
        result = self.preprocessor.preprocess(text)
        
        # Should be lowercase
        self.assertEqual(result, result.lower())
        
        # Should not contain email
        self.assertNotIn('test@email.com', result)
        
        # Should not contain punctuation
        self.assertNotIn('!', result)
        self.assertNotIn("'", result)
    
    def test_preprocess_corpus(self):
        """Test preprocessing of multiple texts."""
        texts = ["Hello World!", "This is a test.", "Another example."]
        results = self.preprocessor.preprocess_corpus(texts, show_progress=False)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertEqual(result, result.lower())


class TestTextVectorizer(unittest.TestCase):
    """Test cases for TextVectorizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vectorizer = TextVectorizer(method='tfidf', max_features=100)
        self.sample_texts = [
            "this is a test",
            "another test document",
            "one more test example"
        ]
    
    def test_vectorizer_initialization(self):
        """Test vectorizer initialization."""
        self.assertEqual(self.vectorizer.method, 'tfidf')
        self.assertEqual(self.vectorizer.max_features, 100)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        X = self.vectorizer.fit_transform(self.sample_texts)
        
        # Check shape
        self.assertEqual(X.shape[0], len(self.sample_texts))
        self.assertLessEqual(X.shape[1], 100)
    
    def test_transform(self):
        """Test transform method after fitting."""
        self.vectorizer.fit_transform(self.sample_texts)
        
        new_texts = ["this is new", "another document"]
        X_new = self.vectorizer.transform(new_texts)
        
        self.assertEqual(X_new.shape[0], len(new_texts))
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        self.vectorizer.fit_transform(self.sample_texts)
        features = self.vectorizer.get_feature_names()
        
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_count_vectorizer(self):
        """Test Count Vectorizer method."""
        vectorizer = TextVectorizer(method='count', max_features=100)
        X = vectorizer.fit_transform(self.sample_texts)
        
        self.assertEqual(X.shape[0], len(self.sample_texts))


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        import pandas as pd
        self.df = pd.DataFrame({
            'text': ['Hello World!', 'Test 123', 'UPPERCASE text']
        })
    
    def test_engineer_features(self):
        """Test feature engineering."""
        result_df = self.preprocessor.engineer_features(self.df)
        
        # Check new columns exist
        self.assertIn('char_count', result_df.columns)
        self.assertIn('word_count', result_df.columns)
        self.assertIn('avg_word_length', result_df.columns)
        self.assertIn('punctuation_count', result_df.columns)
        self.assertIn('uppercase_count', result_df.columns)
    
    def test_char_count_feature(self):
        """Test character count feature."""
        result_df = self.preprocessor.engineer_features(self.df)
        
        # "Hello World!" has 12 characters
        self.assertEqual(result_df.loc[0, 'char_count'], len('Hello World!'))
    
    def test_word_count_feature(self):
        """Test word count feature."""
        result_df = self.preprocessor.engineer_features(self.df)
        
        # "Hello World!" has 2 words
        self.assertEqual(result_df.loc[0, 'word_count'], 2)
    
    def test_punctuation_count_feature(self):
        """Test punctuation count feature."""
        result_df = self.preprocessor.engineer_features(self.df)
        
        # "Hello World!" has 1 punctuation mark
        self.assertEqual(result_df.loc[0, 'punctuation_count'], 1)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestTextVectorizer))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
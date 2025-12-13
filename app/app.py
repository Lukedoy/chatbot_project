"""
app/app.py
Flask web application for the chatbot
Grade-3: Package model into application
Grade-5: Professional presentation with explanations
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from src.preprocessor import TextPreprocessor, TextVectorizer
from src.models import LogisticRegressionModel, RandomForestModel, XGBoostModel, LSTMModel
import pickle
import json
import time
from datetime import datetime

app = Flask(__name__)

# Global variables for model and artifacts
model = None
vectorizer = None
label_encoder = None
preprocessor = None
model_info = {}


def load_model_artifacts():
    """Load the best model and required artifacts."""
    global model, vectorizer, label_encoder, preprocessor, model_info
    
    print("🚀 Loading model artifacts...")
    
    try:
        # Check if comparison file exists
        comparison_path = os.path.join(parent_dir, 'models', 'model_comparison.csv')
        if not os.path.exists(comparison_path):
            print("⚠️ model_comparison.csv not found. Using XGBoost as default.")
            best_model_name = 'xgboost'
        else:
            # Load comparison results to find best model
            import pandas as pd
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            
            # Convert to numeric
            for col in comparison_df.columns:
                comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
            
            best_model_name = comparison_df['f1_score'].idxmax()
        
        print(f"📦 Best model: {best_model_name}")
    except Exception as e:
        print(f"⚠️ Error loading comparison: {e}")
        print("Using XGBoost as default model")
        best_model_name = 'xgboost'
    
    # Load preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Load vectorizer
    vectorizer = TextVectorizer()
    vectorizer_path = os.path.join(parent_dir, 'models', 'vectorizer.pkl')
    vectorizer.load(vectorizer_path)
    
    # Load label encoder
    encoder_path = os.path.join(parent_dir, 'models', 'label_encoder.pkl')
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load the best model
    try:
        models_dir = os.path.join(parent_dir, 'models')
        
        if best_model_name == 'logistic':
            model = LogisticRegressionModel()
            model.load(os.path.join(models_dir, 'logistic_model.pkl'))
        elif best_model_name == 'random_forest':
            model = RandomForestModel()
            model.load(os.path.join(models_dir, 'random_forest_model.pkl'))
        elif best_model_name == 'xgboost':
            model = XGBoostModel()
            model.load(os.path.join(models_dir, 'xgboost_model.pkl'))
        elif best_model_name == 'lstm':
            model = LSTMModel()
            model.load(
                os.path.join(models_dir, 'lstm_model.h5'),
                os.path.join(models_dir, 'lstm_tokenizer.pkl')
            )
        
        # Load model info
        if os.path.exists(comparison_path):
            import pandas as pd
            comparison_df = pd.read_csv(comparison_path, index_col=0)
            for col in comparison_df.columns:
                comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
            
            model_info = {
                'name': best_model_name,
                'accuracy': float(comparison_df.loc[best_model_name, 'accuracy']),
                'f1_score': float(comparison_df.loc[best_model_name, 'f1_score']),
                'num_classes': len(label_encoder.classes_)
            }
        else:
            model_info = {
                'name': best_model_name,
                'accuracy': 0.0,
                'f1_score': 0.0,
                'num_classes': len(label_encoder.classes_)
            }
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    print("✅ Model loaded successfully!")
    print(f"   Model: {model_info['name']}")
    print(f"   Accuracy: {model_info['accuracy']:.4f}")
    print(f"   F1-Score: {model_info['f1_score']:.4f}")


@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    
    Returns:
        JSON response with prediction and confidence
    """
    try:
        # Get user message
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                'error': 'No message provided'
            }), 400
        
        # Start timing
        start_time = time.time()
        
        # Preprocess text
        processed_text = preprocessor.preprocess(user_message)
        
        # Make prediction
        if model_info['name'] == 'lstm':
            # LSTM uses text directly
            prediction = model.predict([processed_text])[0]
            probabilities = model.predict_proba([processed_text])[0]
        else:
            # Traditional models use vectorized text
            vectorized = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
        
        # Get prediction time
        prediction_time = time.time() - start_time
        
        # Get intent label
        intent = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_intents = [
            {
                'intent': label_encoder.inverse_transform([idx])[0],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        # Generate response based on intent
        response_text = generate_response(intent)
        
        return jsonify({
            'success': True,
            'intent': intent,
            'confidence': confidence,
            'top_predictions': top_3_intents,
            'response': response_text,
            'prediction_time': prediction_time,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


def generate_response(intent):
    """
    Generate appropriate response based on detected intent.
    
    Args:
        intent: Predicted intent
        
    Returns:
        str: Response message
    """
    # Intent-based responses (customize these for your domain)
    responses = {
        # Add your intent-specific responses here
        # Example:
        # 'activate_my_card': "I can help you activate your card! Please provide your card details.",
        # 'balance': "Let me check your account balance for you.",
    }
    
    # Default response
    default_response = f"I understand you're asking about '{intent}'. How can I help you with this?"
    
    return responses.get(intent, default_response)


@app.route('/model_info')
def get_model_info():
    """Get information about the deployed model."""
    return jsonify(model_info)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Load model on startup
    load_model_artifacts()
    
    # Run app
    print("\n" + "="*70)
    print("🚀 Starting Flask Application")
    print("="*70)
    print(f"🌐 Access the chatbot at: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
app/app.py
Flask web application for the chatbot
Grade-3: Package model into application
Grade-5: Professional presentation with explanations
"""

from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('../src')
from preprocessor import TextPreprocessor, TextVectorizer
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
    
    # Load comparison results to find best model
    with open('../models/model_comparison.csv', 'r') as f:
        import pandas as pd
        comparison_df = pd.read_csv(f, index_col=0)
        best_model_name = comparison_df['f1_score'].idxmax()
    
    print(f"📦 Best model: {best_model_name}")
    
    # Load preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    
    # Load vectorizer
    vectorizer = TextVectorizer()
    vectorizer.load('../models/vectorizer.pkl')
    
    # Load label encoder
    with open('../models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load the best model
    if best_model_name == 'logistic':
        from models import LogisticRegressionModel
        model = LogisticRegressionModel()
        model.load('../models/logistic_model.pkl')
    elif best_model_name == 'random_forest':
        from models import RandomForestModel
        model = RandomForestModel()
        model.load('../models/random_forest_model.pkl')
    elif best_model_name == 'xgboost':
        from models import XGBoostModel
        model = XGBoostModel()
        model.load('../models/xgboost_model.pkl')
    elif best_model_name == 'lstm':
        from models import LSTMModel
        model = LSTMModel()
        model.load('../models/lstm_model.h5', '../models/lstm_tokenizer.pkl')
    
    # Load model info
    model_info = {
        'name': best_model_name,
        'accuracy': float(comparison_df.loc[best_model_name, 'accuracy']),
        'f1_score': float(comparison_df.loc[best_model_name, 'f1_score']),
        'num_classes': len(label_encoder.classes_)
    }
    
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
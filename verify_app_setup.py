"""
verify_app_setup.py
Verify that all required files exist for the Flask app to run
"""

import os
import sys

def verify_app_files():
    """Verify all required files for Flask app."""
    
    print("=" * 70)
    print("VERIFYING FLASK APP SETUP")
    print("=" * 70)
    print()
    
    required_files = {
        'Models Directory': [
            ('models/vectorizer.pkl', 'Vectorizer'),
            ('models/label_encoder.pkl', 'Label Encoder'),
            ('models/model_comparison.csv', 'Model Comparison'),
        ],
        'At least one trained model': [
            ('models/logistic_model.pkl', 'Logistic Regression', False),
            ('models/random_forest_model.pkl', 'Random Forest', False),
            ('models/xgboost_model.pkl', 'XGBoost', False),
            ('models/lstm_model.h5', 'LSTM', False),
        ],
        'Source Code': [
            ('src/preprocessor.py', 'Preprocessor'),
            ('src/models.py', 'Models'),
        ],
        'Flask App': [
            ('app/app.py', 'Flask App'),
            ('app/templates/index.html', 'HTML Template'),
        ]
    }
    
    all_good = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        if category == 'At least one trained model':
            # Check if at least one model exists
            found_model = False
            for file_path, name, _ in files:
                exists = os.path.exists(file_path)
                if exists:
                    print(f"  ✅ {name}: {file_path}")
                    found_model = True
                else:
                    print(f"  ⚠️  {name}: {file_path} (not found)")
            
            if not found_model:
                print(f"\n  ❌ ERROR: No trained models found!")
                all_good = False
        else:
            # All files in this category are required
            for item in files:
                if len(item) == 2:
                    file_path, name = item
                    required = True
                else:
                    file_path, name, required = item
                
                exists = os.path.exists(file_path)
                
                if exists:
                    print(f"  ✅ {name}: {file_path}")
                elif required:
                    print(f"  ❌ {name}: {file_path} (MISSING - REQUIRED)")
                    all_good = False
                else:
                    print(f"  ⚠️  {name}: {file_path} (optional)")
    
    print("\n" + "=" * 70)
    
    if all_good:
        print("✅ ALL REQUIRED FILES PRESENT")
        print("=" * 70)
        print("\nYou can now run the Flask app:")
        print("   cd app")
        print("   python app.py")
        print("\nThen open: http://localhost:5000")
        return True
    else:
        print("❌ MISSING REQUIRED FILES")
        print("=" * 70)
        print("\nTo fix:")
        print("1. Run: python fix_missing_files.py")
        print("2. If models are missing, run: python run_pipeline.py")
        return False


if __name__ == "__main__":
    success = verify_app_files()
    sys.exit(0 if success else 1)
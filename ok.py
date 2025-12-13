# quick_create_encoder.py
from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create models dir
os.makedirs('models', exist_ok=True)

# Load data
dataset = load_dataset("banking77")
df = pd.DataFrame(dataset['train'])

# Create and save label encoder
label_encoder = LabelEncoder()
label_encoder.fit(df['label'])

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"✅ Created label_encoder.pkl with {len(label_encoder.classes_)} classes")
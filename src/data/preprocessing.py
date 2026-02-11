import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from ..config import MODELS_DIR

def preprocess_data(df):
    # Handle missing values (if any)
    df = df.dropna()
    
    # Encode Categorical Variables
    le_dict = {}
    categorical_cols = ['gender', 'education', 'industry', 'location']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Save label encoders for future inference
    joblib.dump(le_dict, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    
    return df

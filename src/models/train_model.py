import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from ..config import MODELS_DIR, RANDOM_STATE

def train_models(df):
    features = ['gender_encoded', 'age', 'education_encoded', 'industry_encoded', 'years_experience', 'location_encoded']
    target = 'salary'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = {}
    best_model = None
    best_score = -float('inf')
    
    print("Model Evaluation Results:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"\n{name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2 Score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            
    # Save best model
    if best_model:
        joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_salary_model.pkl'))
        print(f"\nBest model saved to {os.path.join(MODELS_DIR, 'best_salary_model.pkl')}")

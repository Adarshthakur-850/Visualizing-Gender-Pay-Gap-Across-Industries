import pandas as pd
import numpy as np
import os
from ..config import DATA_RAW, RANDOM_STATE

def load_data(n_samples=1000):
    np.random.seed(RANDOM_STATE)
    
    genders = ['Male', 'Female']
    industries = ['Tech', 'Finance', 'Healthcare', 'Education', 'Retail']
    educations = ['High School', 'Bachelor', 'Master', 'PhD']
    locations = ['New York', 'San Francisco', 'London', 'Remote', 'Bangalore']
    
    data = {
        'employee_id': np.arange(n_samples),
        'gender': np.random.choice(genders, n_samples),
        'age': np.random.randint(22, 60, n_samples),
        'education': np.random.choice(educations, n_samples),
        'industry': np.random.choice(industries, n_samples),
        'years_experience': np.random.randint(0, 40, n_samples),
        'location': np.random.choice(locations, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Simulate Salary with bias for demonstration
    base_salary = 50000
    df['salary'] = base_salary + (df['years_experience'] * 2000)
    
    # Add Industry variance
    industry_mult = {'Tech': 1.5, 'Finance': 1.4, 'Healthcare': 1.2, 'Education': 0.9, 'Retail': 0.8}
    df['salary'] = df['salary'] * df['industry'].map(industry_mult)
    
    # Add Education variance
    edu_mult = {'High School': 1.0, 'Bachelor': 1.2, 'Master': 1.4, 'PhD': 1.6}
    df['salary'] = df['salary'] * df['education'].map(edu_mult)
    
    # Add a gender gap for demonstration (Simulated Pay Gap)
    gender_mult = {'Male': 1.1, 'Female': 1.0}
    df['salary'] = df['salary'] * df['gender'].map(gender_mult)
    
    # Add noise
    df['salary'] = df['salary'] + np.random.normal(0, 5000, n_samples)
    
    file_path = os.path.join(DATA_RAW, 'salary_data.csv')
    df.to_csv(file_path, index=False)
    return df

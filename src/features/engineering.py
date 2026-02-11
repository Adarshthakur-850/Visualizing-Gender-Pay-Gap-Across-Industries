import pandas as pd

def engineer_features(df):
    # Salary per year of experience
    df['salary_per_year_exp'] = df['salary'] / (df['years_experience'] + 1)
    
    return df

from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import engineer_features
from src.visualization.visualize import plot_gender_pay_gap, plot_salary_distribution, plot_correlation_heatmap
from src.models.train_model import train_models

def main():
    print("Starting Gender Pay Gap Analysis...")
    
    # 1. Load Data
    print("1. Loading Data...")
    df = load_data()
    
    # 2. Preprocessing
    print("2. Preprocessing Data...")
    df = preprocess_data(df)
    
    # 3. Feature Engineering
    print("3. Feature Engineering...")
    df = engineer_features(df)
    
    # 4. Visualization
    print("4. Generating Visualizations...")
    plot_gender_pay_gap(df)
    plot_salary_distribution(df)
    plot_correlation_heatmap(df)
    
    # 5. Modeling
    print("5. Training Models...")
    train_models(df)
    
    print("Analysis Completed Successfully!")

if __name__ == "__main__":
    main()

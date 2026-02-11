import matplotlib.pyplot as plt
import seaborn as sns
import os
from ..config import PLOTS_DIR

def plot_gender_pay_gap(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='industry', y='salary', hue='gender', data=df, estimator='mean', errorbar=None)
    plt.title('Average Salary by Industry and Gender')
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gender_pay_gap_industry.png'))
    plt.close()

def plot_salary_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='industry', y='salary', hue='gender', data=df)
    plt.title('Salary Distribution by Industry and Gender')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'salary_distribution.png'))
    plt.close()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    # Select numeric columns including encoded ones
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

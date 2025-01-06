import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_summary_statistics(df, target_columns):
    """
    Generates summary statistics for specified columns.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_columns (list): List of columns to generate statistics for.

    Returns:
        pd.DataFrame: Summary statistics.
    """
    stats = df[target_columns].describe()
    print(f"Summary statistics:\n{stats}")
    return stats

def plot_correlation_heatmap(df, city_name):
    """
    Generates a heatmap of correlations between numerical columns.

    Parameters:
        df (pd.DataFrame): The dataset.
        city_name (str): The name of the city (for plot title).
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'{city_name} - Correlation Heatmap')
    plt.show()

def plot_uhi_boxplots(data, columns, city_name):
    """
    Generate box plots for UHI-related metrics (e.g., temperature, humidity, wind speed).
    """
    plt.figure(figsize=(16, 6))
    for i, column in enumerate(columns):
        plt.subplot(1, len(columns), i + 1)
        sns.boxplot(y=data[column], showfliers=True, flierprops=dict(markerfacecolor='r', marker='o'))
        plt.title(f'{city_name} - {column}')
    plt.tight_layout()
    plt.show()

def plot_day_night_temperature(data, city_name):
    """
    Plot average daytime and nighttime temperatures for a city.
    """
    data['Day_Night'] = data['Hour'].apply(lambda x: 'Daytime' if 6 <= x < 18 else 'Nighttime')
    avg_temp = data.groupby('Day_Night')['HourlyDryBulbTemperature'].mean()

    plt.figure(figsize=(6, 4))
    avg_temp.plot(kind='bar', color=['#EF810E', '#095D79'], alpha=0.85)
    plt.title(f'{city_name} - Daytime vs Nighttime Avg Temperatures')
    plt.ylabel('Temperature (°F)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.show()

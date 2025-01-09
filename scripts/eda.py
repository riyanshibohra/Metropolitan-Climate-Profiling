# Import required libraries for data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

# Basic summary statistics function
def generate_summary_statistics(df):
    """
    Generates and prints summary statistics for the given DataFrame.
    """
    print(df.describe())

# Visualization functions for exploratory data analysis

def plot_histograms(df, columns, title="Histograms of Variables"):
    """Plots histograms for specified columns in a compact grid layout."""
    n_cols = 3  
    n_rows = (len(columns) + n_cols - 1) // n_cols 
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten()  
    
    for i, col in enumerate(columns):
        sns.histplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, columns, title="Boxplots of Variables"):
    """
    Plots boxplots for the specified columns in a compact grid layout.
    """
    n_cols = 3 
    n_rows = (len(columns) + n_cols - 1) // n_cols  
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    axes = axes.flatten()  
    
    for i, column in enumerate(columns):
        sns.boxplot(y=df[column], ax=axes[i])
        axes[i].set_title(column)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Time series visualization function
def plot_time_series(df, date_column, value_column, city_name, title="Time Series Plot"):
    """
    Plots a time series for the specified column in the DataFrame.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df[date_column], df[value_column], label=city_name)
    plt.title(f"{title} - {city_name}", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel(value_column)
    plt.grid(True)
    plt.legend()
    plt.show()

# Correlation analysis visualization
def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """
    Plots a compact heatmap showing correlations between variables in the DataFrame.
    """
    plt.figure(figsize=(8, 6))
    corr_matrix = df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Plot heatmap with mask
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .5}
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Temperature comparison function between day and night
def plot_day_night_temperature(dallas, arlington, denton, temperature_column='HourlyDryBulbTemperature'):
    """
    Plots a bar chart comparing daytime and nighttime average temperatures across cities.
    """
    # Helper function to categorize hours into day/night
    def categorize_day_night(hour):
        if 6 <= hour < 18:
            return 'Daytime'
        else:
            return 'Nighttime'

    # Data preparation - add day/night categories
    dallas['Day_Night'] = dallas['Hour'].apply(categorize_day_night)
    arlington['Day_Night'] = arlington['Hour'].apply(categorize_day_night)
    denton['Day_Night'] = denton['Hour'].apply(categorize_day_night)

    # Calculate average temperatures
    dallas_day_night_avg = dallas.groupby('Day_Night')[temperature_column].mean()
    arlington_day_night_avg = arlington.groupby('Day_Night')[temperature_column].mean()
    denton_day_night_avg = denton.groupby('Day_Night')[temperature_column].mean()

    # Print results
    print("Dallas - Day vs Night Avg Temperature:\n", dallas_day_night_avg)
    print("Arlington - Day vs Night Avg Temperature:\n", arlington_day_night_avg)
    print("Denton - Day vs Night Avg Temperature:\n", denton_day_night_avg)

    # Set up bar chart parameters
    bar_width = 0.2
    index = np.arange(3)  # Three cities

    # Create bar chart visualization
    plt.figure(figsize=(7, 5))
    plt.bar(
        index,
        [dallas_day_night_avg['Daytime'], arlington_day_night_avg['Daytime'], denton_day_night_avg['Daytime']],
        bar_width, color='#EF810E', label='Daytime'
    )
    plt.bar(
        index + bar_width,
        [dallas_day_night_avg['Nighttime'], arlington_day_night_avg['Nighttime'], denton_day_night_avg['Nighttime']],
        bar_width, color='#095D79', label='Nighttime'
    )

    # Customize plot appearance
    plt.xlabel('Cities')
    plt.ylabel('Average Temperature (°F)')
    plt.title('Daytime vs Nighttime Average Temperatures Across Cities')
    plt.xticks(index + bar_width / 2, ['Dallas', 'Arlington', 'Denton'])
    plt.legend()

    plt.tight_layout()
    plt.show()

# Day vs Night Analysis Section

# Data preparation function
def prepare_data_for_analysis(df):
    """
    Prepares the dataset for analysis by ensuring necessary columns exist.
    - Converts 'DATE' to datetime.
    - Creates 'Date' (date only) and 'Day_Night' columns.
    """
    def categorize_day_night(hour):
        if 6 <= hour < 18:
            return 'Daytime'
        else:
            return 'Nighttime'

    # Data preprocessing steps
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    if 'Date' not in df.columns:
        df['Date'] = df['DATE'].dt.date
    if 'Day_Night' not in df.columns and 'Hour' in df.columns:
        df['Day_Night'] = df['Hour'].apply(categorize_day_night)
    return df

# Temperature analysis function
def calculate_nightly_temp_drop(df):
    """
    Calculates the nightly temperature drop for the given dataset.
    - Groups data by 'Date' to calculate min and max temperature.
    - Computes the percentage rate of change.
    - Returns monthly average temperature drop.
    """
    # Calculate temperature metrics
    df['low'] = df.groupby('Date')['HourlyDryBulbTemperature'].transform('min')
    df['high'] = df.groupby('Date')['HourlyDryBulbTemperature'].transform('max')
    df['rate_of_change'] = (df['high'] - df['low']) / df['low'] * 100
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    monthly_avg_drop = df.groupby('Month')['rate_of_change'].mean().reset_index()
    return monthly_avg_drop

# Temperature drop visualization function
def plot_temp_drop(dallas_monthly, arlington_monthly, denton_monthly):
    """
    Plots the monthly average nightly temperature drop for Dallas, Arlington, and Denton.
    """
    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(dallas_monthly['Month'], dallas_monthly['rate_of_change'], label='Dallas', marker='o')
    plt.plot(arlington_monthly['Month'], arlington_monthly['rate_of_change'], label='Arlington', marker='s')
    plt.plot(denton_monthly['Month'], denton_monthly['rate_of_change'], label='Denton', marker='^')

    # Customize plot appearance
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(dallas_monthly['Month'], months)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add labels and formatting
    plt.xlabel('Month')
    plt.ylabel('Average Nightly Temp Drop (%)')
    plt.title('Region Comparison - Monthly Avg Drop in Temp at Night')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Main orchestration function
def plot_monthly_avg_temp_drop(dallas, arlington, denton):
    """
    Orchestrates the preparation, calculation, and plotting of monthly avg nightly temp drop.
    """
    # Prepare and process data
    dallas = prepare_data_for_analysis(dallas)
    arlington = prepare_data_for_analysis(arlington)
    denton = prepare_data_for_analysis(denton)

    # Filter for nighttime data
    dallas_night = dallas[dallas['Day_Night'] == 'Nighttime']
    arlington_night = arlington[arlington['Day_Night'] == 'Nighttime']
    denton_night = denton[denton['Day_Night'] == 'Nighttime']

    # Calculate temperature metrics
    dallas_monthly = calculate_nightly_temp_drop(dallas_night)
    arlington_monthly = calculate_nightly_temp_drop(arlington_night)
    denton_monthly = calculate_nightly_temp_drop(denton_night)

    # Create visualization
    plot_temp_drop(dallas_monthly, arlington_monthly, denton_monthly)

# Humidity analysis visualization function
def plot_seasonal_humidity_variation(dallas, arlington, denton):
    """
    Plots the seasonal variation of relative humidity for Dallas, Arlington, and Denton.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Set up plot style
    custom_palette = sns.color_palette("Set2", n_colors=3)
    plt.figure(figsize=(12, 8))

    # Create line plots for each city
    sns.lineplot(
        x=pd.to_datetime(arlington['DATE']).dt.month,
        y=arlington['HourlyRelativeHumidity'],
        label='Arlington',
        color=custom_palette[0],
        linestyle='-',
        marker='o',
        markersize=8,
    )
    sns.lineplot(
        x=pd.to_datetime(dallas['DATE']).dt.month,
        y=dallas['HourlyRelativeHumidity'],
        label='Dallas',
        color=custom_palette[1],
        linestyle='-',
        marker='s',
        markersize=8,
    )
    sns.lineplot(
        x=pd.to_datetime(denton['DATE']).dt.month,
        y=denton['HourlyRelativeHumidity'],
        label='Denton',
        color=custom_palette[2],
        linestyle='-',
        marker='^',
        markersize=8,
    )

    # Add plot labels and styling
    plt.title(
        'Seasonal Variation of Relative Humidity in Different Cities',
        fontsize=18,
        fontweight='bold',
        color='darkblue',
    )
    plt.xlabel('Month', fontsize=14, fontweight='bold', color='darkgreen')
    plt.ylabel('Relative Humidity (%)', fontsize=14, fontweight='bold', color='darkred')
    plt.legend(title='Cities', title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add seasonal background highlights
    seasons = [
        (1, 'Winter', 'lightblue'),
        (4, 'Spring', 'lightgreen'),
        (7, 'Summer', 'lightcoral'),
        (10, 'Fall', 'khaki'),
    ]
    for month, season_name, color in seasons:
        plt.axvspan(month - 1, month + 2, alpha=0.2, color=color, label=season_name)
        plt.text(
            month + 1,
            plt.ylim()[1] * 0.95,
            season_name,
            fontsize=12,
            ha='center',
            va='center',
            color='black',
            fontweight='bold',
        )

    plt.tight_layout()
    plt.show()
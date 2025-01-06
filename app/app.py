import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Urban Heat Island Analyzer",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache data for all cities"""
    try:
        # Load data for all cities
        dallas_data = pd.read_csv('data/Dallas.csv')
        arlington_data = pd.read_csv('data/Arlington.csv')
        denton_data = pd.read_csv('data/Denton.csv')
        
        cities_data = {
            'Dallas': dallas_data,
            'Arlington': arlington_data,
            'Denton': denton_data
        }
        
        # Process each city's data
        for city, df in cities_data.items():
            # Convert DATE to datetime
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Convert temperature columns to numeric
            temp_columns = ['HourlyDryBulbTemperature', 'HourlyWetBulbTemperature', 
                          'HourlyDewPointTemperature']
            for col in temp_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert humidity to numeric
            if 'HourlyRelativeHumidity' in df.columns:
                df['HourlyRelativeHumidity'] = pd.to_numeric(df['HourlyRelativeHumidity'], errors='coerce')
        
        return cities_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_uhi_metrics(data, primary_city, secondary_city, start_date, end_date):
    """Calculate UHI metrics between two cities"""
    # Filter data by date range
    primary_data = data[primary_city][
        (data[primary_city]['DATE'] >= start_date) & 
        (data[primary_city]['DATE'] <= end_date)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    secondary_data = data[secondary_city][
        (data[secondary_city]['DATE'] >= start_date) & 
        (data[secondary_city]['DATE'] <= end_date)
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Calculate basic metrics
    temp_diff = primary_data['HourlyDryBulbTemperature'].mean() - secondary_data['HourlyDryBulbTemperature'].mean()
    max_diff = primary_data['HourlyDryBulbTemperature'].max() - secondary_data['HourlyDryBulbTemperature'].max()
    min_diff = primary_data['HourlyDryBulbTemperature'].min() - secondary_data['HourlyDryBulbTemperature'].min()
    
    # Calculate day/night differences
    primary_data['Hour'] = primary_data['DATE'].dt.hour
    secondary_data['Hour'] = secondary_data['DATE'].dt.hour
    
    # Calculate day and night differences separately for each city
    primary_day = primary_data[primary_data['Hour'].between(6, 18)]['HourlyDryBulbTemperature'].mean()
    primary_night = primary_data[~primary_data['Hour'].between(6, 18)]['HourlyDryBulbTemperature'].mean()
    
    secondary_day = secondary_data[secondary_data['Hour'].between(6, 18)]['HourlyDryBulbTemperature'].mean()
    secondary_night = secondary_data[~secondary_data['Hour'].between(6, 18)]['HourlyDryBulbTemperature'].mean()
    
    day_diff = primary_day - secondary_day
    night_diff = primary_night - secondary_night
    
    return {
        'mean_diff': temp_diff,
        'max_diff': max_diff,
        'min_diff': min_diff,
        'day_diff': day_diff,
        'night_diff': night_diff
    }

def plot_temperature_comparison(data, primary_city, secondary_city, start_date, end_date):
    """Create temperature comparison plot"""
    fig = go.Figure()
    
    # Filter data by date range
    for city, color in [(primary_city, 'red'), (secondary_city, 'blue')]:
        city_data = data[city][
            (data[city]['DATE'] >= start_date) & 
            (data[city]['DATE'] <= end_date)
        ]
        
        fig.add_trace(go.Scatter(
            x=city_data['DATE'],
            y=city_data['HourlyDryBulbTemperature'],
            name=city,
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title=f"Temperature Comparison: {primary_city} vs {secondary_city}",
        xaxis_title="Date",
        yaxis_title="Temperature (°F)",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def main():
    # Title and description
    st.title("Metropolitan Climate Profiling: Urban Heat Island Analysis")
    st.markdown("""
    Analyze and visualize Urban Heat Island (UHI) effects across Dallas, Arlington, and Denton.
    Compare temperature patterns and identify urban heating patterns.
    """)
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check the error messages above.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Controls")
        
        # City selection
        primary_city = st.selectbox("Select Primary City", ["Dallas", "Arlington", "Denton"])
        secondary_city = st.selectbox(
            "Select Comparison City",
            [city for city in ["Dallas", "Arlington", "Denton"] if city != primary_city]
        )
        
        # Date range selection
        st.subheader("Date Range")
        min_date = min(data[primary_city]['DATE'].min(), data[secondary_city]['DATE'].min())
        max_date = max(data[primary_city]['DATE'].max(), data[secondary_city]['DATE'].max())
        
        start_date, end_date = st.date_input(
            "Select period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Convert to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Temperature Analysis", "UHI Intensity", "Detailed Statistics"])
    
    with tab1:
        # Temperature comparison plot
        st.plotly_chart(
            plot_temperature_comparison(data, primary_city, secondary_city, start_date, end_date),
            use_container_width=True
        )
    
    with tab2:
        # UHI Analysis
        metrics = calculate_uhi_metrics(data, primary_city, secondary_city, start_date, end_date)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Temperature Difference",
                f"{abs(metrics['mean_diff']):.1f}°F",
                f"{primary_city} is {'warmer' if metrics['mean_diff'] > 0 else 'cooler'}"
            )
        
        with col2:
            st.metric(
                "Day vs Night Difference",
                f"Day: {metrics['day_diff']:.1f}°F",
                f"Night: {metrics['night_diff']:.1f}°F"
            )
        
        with col3:
            st.metric(
                "Maximum Temperature Difference",
                f"{abs(metrics['max_diff']):.1f}°F",
                f"Min Diff: {metrics['min_diff']:.1f}°F"
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Statistics")
            filtered_primary = data[primary_city][
                (data[primary_city]['DATE'] >= start_date) & 
                (data[primary_city]['DATE'] <= end_date)
            ]
            filtered_secondary = data[secondary_city][
                (data[secondary_city]['DATE'] >= start_date) & 
                (data[secondary_city]['DATE'] <= end_date)
            ]
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Maximum', 'Minimum', 'Std Dev'],
                primary_city: [
                    f"{filtered_primary['HourlyDryBulbTemperature'].mean():.1f}°F",
                    f"{filtered_primary['HourlyDryBulbTemperature'].max():.1f}°F",
                    f"{filtered_primary['HourlyDryBulbTemperature'].min():.1f}°F",
                    f"{filtered_primary['HourlyDryBulbTemperature'].std():.1f}°F"
                ],
                secondary_city: [
                    f"{filtered_secondary['HourlyDryBulbTemperature'].mean():.1f}°F",
                    f"{filtered_secondary['HourlyDryBulbTemperature'].max():.1f}°F",
                    f"{filtered_secondary['HourlyDryBulbTemperature'].min():.1f}°F",
                    f"{filtered_secondary['HourlyDryBulbTemperature'].std():.1f}°F"
                ]
            })
            st.table(stats_df)
        
        with col2:
            st.subheader("Hourly Pattern")
            # Add hourly temperature pattern plot
            for city, color in [(primary_city, 'red'), (secondary_city, 'blue')]:
                city_data = data[city][
                    (data[city]['DATE'] >= start_date) & 
                    (data[city]['DATE'] <= end_date)
                ]
                city_data['Hour'] = city_data['DATE'].dt.hour
                hourly_temps = city_data.groupby('Hour')['HourlyDryBulbTemperature'].mean()
                
                fig = px.line(
                    x=hourly_temps.index,
                    y=hourly_temps.values,
                    labels={'x': 'Hour of Day', 'y': 'Average Temperature (°F)'},
                    title=f"Average Hourly Temperature Pattern"
                )
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Simple COVID-19 Data Analysis & Forecasting
Works with basic packages: pandas, matplotlib, seaborn, requests, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

warnings.filterwarnings('ignore')

print("🚀 Starting Simple COVID-19 Data Analysis...")
print("=" * 50)

# Set up directories
plots_dir = "plots"
output_dir = "output"
data_dir = "data"

for directory in [plots_dir, output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Step 1: Dataset Loading
print("\n📊 STEP 1: Dataset Loading")
print("-" * 30)

# Use existing dataset if available, otherwise download
local_filename = os.path.join(data_dir, "covid_data.csv")
owid_filename = os.path.join(data_dir, "owid-covid-data.csv")

# Check which dataset to use
if os.path.exists(local_filename):
    print("✅ Using existing covid_data.csv")
    df = pd.read_csv(local_filename)
elif os.path.exists(owid_filename):
    print("✅ Using existing owid-covid-data.csv")
    df = pd.read_csv(owid_filename)
else:
    print("🌐 Downloading COVID-19 dataset...")
    try:
        owid_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        response = requests.get(owid_url)
        response.raise_for_status()
        with open(owid_filename, 'wb') as file:
            file.write(response.content)
        df = pd.read_csv(owid_filename)
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        exit(1)

print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"🌍 Countries: {df['location'].nunique()}")

# Step 2: Data Preprocessing
print("\n🔧 STEP 2: Data Preprocessing")
print("-" * 30)

# Remove aggregate rows if they exist
if 'iso_code' in df.columns:
    df_countries = df[~df['iso_code'].str.startswith('OWID_', na=False)].copy()
    print(f"📊 Removed aggregate rows: {len(df) - len(df_countries)} rows")
else:
    df_countries = df.copy()

# Convert date column
df_countries['date'] = pd.to_datetime(df_countries['date'])

# Handle missing values
numeric_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
for col in numeric_columns:
    if col in df_countries.columns:
        df_countries[col] = df_countries[col].fillna(0)

# Calculate key metrics
df_countries['case_fatality_rate'] = np.where(
    df_countries['total_cases'] > 0,
    (df_countries['total_deaths'] / df_countries['total_cases']) * 100,
    0
)

print("✅ Data preprocessing completed!")

# Step 3: Exploratory Data Analysis
print("\n📊 STEP 3: Exploratory Data Analysis")
print("-" * 30)

# Global trends
global_daily = df_countries.groupby('date').agg({
    'new_cases': 'sum',
    'new_deaths': 'sum',
    'total_cases': 'sum',
    'total_deaths': 'sum'
}).reset_index()

# Calculate 7-day averages
global_daily['new_cases_7day_avg'] = global_daily['new_cases'].rolling(7).mean()
global_daily['new_deaths_7day_avg'] = global_daily['new_deaths'].rolling(7).mean()

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🌍 Global COVID-19 Trends', fontsize=16, fontweight='bold')

# New cases with 7-day average
ax1.plot(global_daily['date'], global_daily['new_cases'], 'lightblue', alpha=0.3, label='Daily')
ax1.plot(global_daily['date'], global_daily['new_cases_7day_avg'], 'blue', linewidth=2, label='7-Day Avg')
ax1.set_title('📈 Daily New Cases')
ax1.set_ylabel('Cases')
ax1.legend()
ax1.grid(True, alpha=0.3)

# New deaths with 7-day average
ax2.plot(global_daily['date'], global_daily['new_deaths'], 'lightcoral', alpha=0.3, label='Daily')
ax2.plot(global_daily['date'], global_daily['new_deaths_7day_avg'], 'red', linewidth=2, label='7-Day Avg')
ax2.set_title('💀 Daily New Deaths')
ax2.set_ylabel('Deaths')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Cumulative cases
ax3.plot(global_daily['date'], global_daily['total_cases'], 'darkblue', linewidth=2)
ax3.set_title('📊 Cumulative Cases')
ax3.set_ylabel('Total Cases')
ax3.grid(True, alpha=0.3)

# Cumulative deaths
ax4.plot(global_daily['date'], global_daily['total_deaths'], 'darkred', linewidth=2)
ax4.set_title('💀 Cumulative Deaths')
ax4.set_ylabel('Total Deaths')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "global_trends.png"), dpi=300, bbox_inches='tight')
plt.show()

# Top countries analysis
latest_data = df_countries[df_countries['date'] == df_countries['date'].max()]
top_10_cases = latest_data.nlargest(10, 'total_cases')

plt.figure(figsize=(12, 8))
plt.barh(top_10_cases['location'], top_10_cases['total_cases'], color='skyblue')
plt.title('🏆 Top 10 Countries by Total Cases')
plt.xlabel('Total Cases')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top_countries.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"📊 EDA completed!")
print(f"📈 Peak daily cases: {global_daily['new_cases'].max():,}")
print(f"💀 Peak daily deaths: {global_daily['new_deaths'].max():,}")
print(f"📊 Total cases to date: {global_daily['total_cases'].max():,}")
print(f"💀 Total deaths to date: {global_daily['total_deaths'].max():,}")

# Step 4: Statistical Analysis
print("\n📈 STEP 4: Statistical Analysis")
print("-" * 30)

# Correlation analysis
numeric_cols = ['total_cases', 'total_deaths', 'case_fatality_rate']
correlation_data = latest_data[numeric_cols].dropna()
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
plt.title('🔗 Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

print("✅ Statistical analysis completed!")

# Step 5: Simple Forecasting
print("\n🤖 STEP 5: Simple Forecasting")
print("-" * 30)

# Prepare data for forecasting
forecast_data = global_daily[['date', 'new_cases']].copy()
forecast_data['days_since_start'] = (forecast_data['date'] - forecast_data['date'].min()).dt.days

# Train-test split
train_size = int(len(forecast_data) * 0.8)
train_data = forecast_data.iloc[:train_size]
test_data = forecast_data.iloc[train_size:]

# Linear Regression model
X_train = train_data[['days_since_start']]
y_train = train_data['new_cases']
X_test = test_data[['days_since_start']]
y_test = test_data['new_cases']

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"   RMSE: {rmse:.0f}")
print(f"   MAE: {mae:.0f}")
print(f"   R²: {r2:.3f}")

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test, 'b-', label='Actual', linewidth=2)
plt.plot(test_data['date'], y_pred, 'r--', label='Predicted', linewidth=2)
plt.title('🤖 Linear Regression Forecast')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "forecast.png"), dpi=300, bbox_inches='tight')
plt.show()

# Generate 30-day forecast
last_day = forecast_data['days_since_start'].max()
future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
future_forecast = lr_model.predict(future_days)

plt.figure(figsize=(12, 6))
plt.plot(forecast_data['date'], forecast_data['new_cases'], 'b-', label='Historical', alpha=0.7)
future_dates = pd.date_range(start=forecast_data['date'].max(), periods=30, freq='D')
plt.plot(future_dates, future_forecast, 'r--', label='30-Day Forecast', linewidth=2)
plt.title('🔮 30-Day Forecast')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "30day_forecast.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"🔮 30-day forecast average: {np.mean(future_forecast):.0f} cases/day")

# Step 6: Save Results
print("\n💾 STEP 6: Save Results")
print("-" * 30)

# Save summary statistics
summary_stats = {
    'total_countries': df_countries['location'].nunique(),
    'date_range': f"{df_countries['date'].min()} to {df_countries['date'].max()}",
    'total_cases': int(global_daily['total_cases'].max()),
    'total_deaths': int(global_daily['total_deaths'].max()),
    'peak_daily_cases': int(global_daily['new_cases'].max()),
    'peak_daily_deaths': int(global_daily['new_deaths'].max()),
    'model_rmse': float(rmse),
    'model_mae': float(mae),
    'model_r2': float(r2),
    'forecast_avg': float(np.mean(future_forecast))
}

with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
    json.dump(summary_stats, f, indent=2)

# Save processed data
df_countries.to_csv(os.path.join(output_dir, 'processed_covid_data.csv'), index=False)

print("✅ Results saved!")

# Additional Analysis: Continental Comparison
print("\n🌍 STEP 7: Continental Analysis")
print("-" * 30)

if 'continent' in df_countries.columns:
    continent_data = df_countries.groupby('continent').agg({
        'total_cases': 'sum',
        'total_deaths': 'sum',
        'case_fatality_rate': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(continent_data['continent'], continent_data['total_cases'], color='lightblue')
    plt.title('🌍 Total Cases by Continent')
    plt.ylabel('Total Cases')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "continent_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Continental analysis completed!")

# Monthly Analysis
print("\n📅 STEP 8: Monthly Analysis")
print("-" * 30)

global_daily['month'] = global_daily['date'].dt.month
global_daily['year'] = global_daily['date'].dt.year

monthly_data = global_daily.groupby(['year', 'month']).agg({
    'new_cases': 'sum',
    'new_deaths': 'sum'
}).reset_index()

plt.figure(figsize=(12, 6))
for year in monthly_data['year'].unique():
    year_data = monthly_data[monthly_data['year'] == year]
    plt.plot(year_data['month'], year_data['new_cases'], marker='o', label=str(year))

plt.title('📅 Monthly New Cases by Year')
plt.xlabel('Month')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "monthly_trends.png"), dpi=300, bbox_inches='tight')
plt.show()

print("✅ Monthly analysis completed!")

# Final Summary
print("\n" + "=" * 50)
print("🎉 COVID-19 Data Analysis Complete!")
print("=" * 50)
print("📊 Key Results:")
print(f"   🌍 Countries Analyzed: {summary_stats['total_countries']}")
print(f"   📅 Analysis Period: {summary_stats['date_range']}")
print(f"   📈 Total Cases: {summary_stats['total_cases']:,}")
print(f"   💀 Total Deaths: {summary_stats['total_deaths']:,}")
print(f"   📊 Peak Daily Cases: {summary_stats['peak_daily_cases']:,}")
print(f"   💀 Peak Daily Deaths: {summary_stats['peak_daily_deaths']:,}")
print(f"   🤖 Model R²: {summary_stats['model_r2']:.3f}")
print(f"   🔮 Forecast Average: {summary_stats['forecast_avg']:.0f} cases/day")
print("\n📁 Files Created:")
print(f"   📊 Plots saved in: {plots_dir}/")
print(f"   📋 Results saved in: {output_dir}/")
print(f"   📈 Visualizations: 8 comprehensive charts")
print(f"   🤖 Simple forecasting model implemented")
print(f"   📊 Processed data saved for further analysis")
print("\n🎯 Analysis completed successfully!")
print("📚 Ready for presentation and further exploration!")

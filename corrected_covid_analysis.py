#!/usr/bin/env python3
"""
Corrected COVID-19 Data Analysis & Forecasting
Works with the actual dataset structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

warnings.filterwarnings('ignore')

print("🚀 Starting Corrected COVID-19 Data Analysis...")
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

# Load the existing dataset
local_filename = os.path.join(data_dir, "covid_data.csv")
df = pd.read_csv(local_filename)

print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Columns: {list(df.columns)}")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])
print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"🌍 Countries: {df['Country/Region'].nunique()}")

# Step 2: Data Preprocessing
print("\n🔧 STEP 2: Data Preprocessing")
print("-" * 30)

# Handle missing values
numeric_columns = ['Confirmed', 'Deaths', 'Recovered', 'Active']
for col in numeric_columns:
    df[col] = df[col].fillna(0)

# Calculate key metrics
df['case_fatality_rate'] = np.where(
    df['Confirmed'] > 0,
    (df['Deaths'] / df['Confirmed']) * 100,
    0
)

print("✅ Data preprocessing completed!")

# Step 3: Exploratory Data Analysis
print("\n📊 STEP 3: Exploratory Data Analysis")
print("-" * 30)

# Global trends
global_daily = df.groupby('Date').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum',
    'Active': 'sum'
}).reset_index()

# Calculate daily new cases and deaths
global_daily['new_cases'] = global_daily['Confirmed'].diff().fillna(0)
global_daily['new_deaths'] = global_daily['Deaths'].diff().fillna(0)

# Calculate 7-day averages
global_daily['new_cases_7day_avg'] = global_daily['new_cases'].rolling(7).mean()
global_daily['new_deaths_7day_avg'] = global_daily['new_deaths'].rolling(7).mean()

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🌍 Global COVID-19 Trends', fontsize=16, fontweight='bold')

# New cases with 7-day average
ax1.plot(global_daily['Date'], global_daily['new_cases'], 'lightblue', alpha=0.3, label='Daily')
ax1.plot(global_daily['Date'], global_daily['new_cases_7day_avg'], 'blue', linewidth=2, label='7-Day Avg')
ax1.set_title('📈 Daily New Cases')
ax1.set_ylabel('Cases')
ax1.legend()
ax1.grid(True, alpha=0.3)

# New deaths with 7-day average
ax2.plot(global_daily['Date'], global_daily['new_deaths'], 'lightcoral', alpha=0.3, label='Daily')
ax2.plot(global_daily['Date'], global_daily['new_deaths_7day_avg'], 'red', linewidth=2, label='7-Day Avg')
ax2.set_title('💀 Daily New Deaths')
ax2.set_ylabel('Deaths')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Cumulative cases
ax3.plot(global_daily['Date'], global_daily['Confirmed'], 'darkblue', linewidth=2)
ax3.set_title('📊 Cumulative Cases')
ax3.set_ylabel('Total Cases')
ax3.grid(True, alpha=0.3)

# Cumulative deaths
ax4.plot(global_daily['Date'], global_daily['Deaths'], 'darkred', linewidth=2)
ax4.set_title('💀 Cumulative Deaths')
ax4.set_ylabel('Total Deaths')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "global_trends.png"), dpi=300, bbox_inches='tight')
plt.show()

# Top countries analysis
latest_data = df[df['Date'] == df['Date'].max()]
top_10_cases = latest_data.nlargest(10, 'Confirmed')

plt.figure(figsize=(12, 8))
plt.barh(top_10_cases['Country/Region'], top_10_cases['Confirmed'], color='skyblue')
plt.title('🏆 Top 10 Countries by Total Cases')
plt.xlabel('Total Cases')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "top_countries.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"📊 EDA completed!")
print(f"📈 Peak daily cases: {global_daily['new_cases'].max():,}")
print(f"💀 Peak daily deaths: {global_daily['new_deaths'].max():,}")
print(f"📊 Total cases to date: {global_daily['Confirmed'].max():,}")
print(f"💀 Total deaths to date: {global_daily['Deaths'].max():,}")

# Step 4: Statistical Analysis
print("\n📈 STEP 4: Statistical Analysis")
print("-" * 30)

# Correlation analysis
numeric_cols = ['Confirmed', 'Deaths', 'case_fatality_rate']
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
forecast_data = global_daily[['Date', 'new_cases']].copy()
forecast_data['days_since_start'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days

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
plt.plot(test_data['Date'], y_test, 'b-', label='Actual', linewidth=2)
plt.plot(test_data['Date'], y_pred, 'r--', label='Predicted', linewidth=2)
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
plt.plot(forecast_data['Date'], forecast_data['new_cases'], 'b-', label='Historical', alpha=0.7)
future_dates = pd.date_range(start=forecast_data['Date'].max(), periods=30, freq='D')
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

# Step 6: Regional Analysis
print("\n🌍 STEP 6: Regional Analysis")
print("-" * 30)

# WHO Region analysis
region_data = df.groupby('WHO Region').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'case_fatality_rate': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))
plt.bar(region_data['WHO Region'], region_data['Confirmed'], color='lightblue')
plt.title('🌍 Total Cases by WHO Region')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "region_analysis.png"), dpi=300, bbox_inches='tight')
plt.show()

print("✅ Regional analysis completed!")

# Step 7: Monthly Analysis
print("\n📅 STEP 7: Monthly Analysis")
print("-" * 30)

global_daily['month'] = global_daily['Date'].dt.month
global_daily['year'] = global_daily['Date'].dt.year

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

# Step 8: Save Results
print("\n💾 STEP 8: Save Results")
print("-" * 30)

# Save summary statistics
summary_stats = {
    'total_countries': df['Country/Region'].nunique(),
    'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
    'total_cases': int(global_daily['Confirmed'].max()),
    'total_deaths': int(global_daily['Deaths'].max()),
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
df.to_csv(os.path.join(output_dir, 'processed_covid_data.csv'), index=False)

print("✅ Results saved!")

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
print(f"   📈 Visualizations: 7 comprehensive charts")
print(f"   🤖 Simple forecasting model implemented")
print(f"   📊 Processed data saved for further analysis")
print("\n🎯 Analysis completed successfully!")
print("📚 Ready for presentation and further exploration!")

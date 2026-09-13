#!/usr/bin/env python3
"""
Fast COVID-19 Data Analysis - Quick Execution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

print("🚀 Fast COVID-19 Analysis Starting...")
print("=" * 40)

# Setup
plots_dir = "plots"
output_dir = "output"
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv("data/covid_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
print(f"✅ Loaded: {df.shape[0]} rows, {df['Country/Region'].nunique()} countries")

# Global analysis
print("\n📈 Global analysis...")
global_data = df.groupby('Date').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum'
}).reset_index()

global_data['new_cases'] = global_data['Confirmed'].diff().fillna(0)
global_data['new_deaths'] = global_data['Deaths'].diff().fillna(0)

# Quick visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(global_data['Date'], global_data['new_cases'].rolling(7).mean())
plt.title('📈 Daily New Cases (7-day avg)')
plt.ylabel('Cases')

plt.subplot(2, 2, 2)
plt.plot(global_data['Date'], global_data['new_deaths'].rolling(7).mean(), 'red')
plt.title('💀 Daily New Deaths (7-day avg)')
plt.ylabel('Deaths')

plt.subplot(2, 2, 3)
plt.plot(global_data['Date'], global_data['Confirmed'], 'blue')
plt.title('📊 Cumulative Cases')
plt.ylabel('Total Cases')

plt.subplot(2, 2, 4)
plt.plot(global_data['Date'], global_data['Deaths'], 'darkred')
plt.title('💀 Cumulative Deaths')
plt.ylabel('Total Deaths')

plt.tight_layout()
plt.savefig(f"{plots_dir}/quick_analysis.png", dpi=150)
plt.show()

# Top countries
print("\n🏆 Top countries analysis...")
latest = df[df['Date'] == df['Date'].max()]
top_10 = latest.nlargest(10, 'Confirmed')

plt.figure(figsize=(10, 6))
plt.barh(top_10['Country/Region'], top_10['Confirmed'])
plt.title('🏆 Top 10 Countries by Cases')
plt.xlabel('Total Cases')
plt.tight_layout()
plt.savefig(f"{plots_dir}/top_countries.png", dpi=150)
plt.show()

# Simple forecasting
print("\n🤖 Quick forecasting...")
forecast_data = global_data[['Date', 'new_cases']].copy()
forecast_data['days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days

# Train model (80/20 split)
split_idx = int(len(forecast_data) * 0.8)
train = forecast_data.iloc[:split_idx]
test = forecast_data.iloc[split_idx:]

model = LinearRegression()
model.fit(train[['days']], train['new_cases'])
preds = model.predict(test[['days']])

# Metrics
rmse = np.sqrt(mean_squared_error(test['new_cases'], preds))
r2 = r2_score(test['new_cases'], preds)

print(f"📊 Model Performance: R² = {r2:.3f}, RMSE = {rmse:.0f}")

# 30-day forecast
future_days = np.arange(forecast_data['days'].max() + 1, forecast_data['days'].max() + 31).reshape(-1, 1)
future_forecast = model.predict(future_days)

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(forecast_data['Date'], forecast_data['new_cases'], 'b-', alpha=0.7, label='Historical')
future_dates = pd.date_range(start=forecast_data['Date'].max(), periods=30, freq='D')
plt.plot(future_dates, future_forecast, 'r--', linewidth=2, label='30-Day Forecast')
plt.title('🔮 30-Day Forecast')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{plots_dir}/forecast.png", dpi=150)
plt.show()

# Save results
print("\n💾 Saving results...")
results = {
    'countries': df['Country/Region'].nunique(),
    'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
    'total_cases': int(global_data['Confirmed'].max()),
    'total_deaths': int(global_data['Deaths'].max()),
    'peak_cases': int(global_data['new_cases'].max()),
    'peak_deaths': int(global_data['new_deaths'].max()),
    'model_r2': float(r2),
    'forecast_avg': float(np.mean(future_forecast))
}

with open(f"{output_dir}/quick_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 40)
print("🎉 Fast Analysis Complete!")
print("=" * 40)
print(f"🌍 Countries: {results['countries']}")
print(f"📈 Total Cases: {results['total_cases']:,}")
print(f"💀 Total Deaths: {results['total_deaths']:,}")
print(f"📊 Peak Cases: {results['peak_cases']:,}")
print(f"🤖 Model R²: {results['model_r2']:.3f}")
print(f"🔮 Forecast Avg: {results['forecast_avg']:.0f}/day")
print(f"\n📁 Files: {len(os.listdir(plots_dir))} plots, results saved")
print("✅ Done!")

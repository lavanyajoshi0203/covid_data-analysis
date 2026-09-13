#!/usr/bin/env python3
"""
COVID-19 Analysis with Return Results
Returns structured data for easy consumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def run_covid_analysis():
    """Main analysis function that returns results"""
    
    print("🚀 Running COVID-19 Analysis...")
    
    # Setup
    plots_dir = "plots"
    output_dir = "output"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv("data/covid_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Global analysis
    global_data = df.groupby('Date').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum'
    }).reset_index()
    
    global_data['new_cases'] = global_data['Confirmed'].diff().fillna(0)
    global_data['new_deaths'] = global_data['Deaths'].diff().fillna(0)
    
    # Top countries
    latest = df[df['Date'] == df['Date'].max()]
    top_10 = latest.nlargest(10, 'Confirmed')
    
    # Simple forecasting
    forecast_data = global_data[['Date', 'new_cases']].copy()
    forecast_data['days'] = (forecast_data['Date'] - forecast_data['Date'].min()).dt.days
    
    split_idx = int(len(forecast_data) * 0.8)
    train = forecast_data.iloc[:split_idx]
    test = forecast_data.iloc[split_idx:]
    
    model = LinearRegression()
    model.fit(train[['days']], train['new_cases'])
    preds = model.predict(test[['days']])
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(test['new_cases'], preds))
    r2 = r2_score(test['new_cases'], preds)
    
    # 30-day forecast
    future_days = np.arange(forecast_data['days'].max() + 1, forecast_data['days'].max() + 31).reshape(-1, 1)
    future_forecast = model.predict(future_days)
    
    # Create results dictionary
    results = {
        'analysis_info': {
            'countries_analyzed': df['Country/Region'].nunique(),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'total_records': len(df),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'global_statistics': {
            'total_cases': int(global_data['Confirmed'].max()),
            'total_deaths': int(global_data['Deaths'].max()),
            'peak_daily_cases': int(global_data['new_cases'].max()),
            'peak_daily_deaths': int(global_data['new_deaths'].max()),
            'case_fatality_rate': round((global_data['Deaths'].max() / global_data['Confirmed'].max()) * 100, 2)
        },
        'top_countries': [
            {
                'country': row['Country/Region'],
                'total_cases': int(row['Confirmed']),
                'total_deaths': int(row['Deaths']),
                'case_fatality_rate': round((row['Deaths'] / row['Confirmed']) * 100, 2) if row['Confirmed'] > 0 else 0
            }
            for _, row in top_10.iterrows()
        ],
        'model_performance': {
            'model_type': 'Linear Regression',
            'r_squared': round(r2, 4),
            'rmse': round(rmse, 2),
            'mae': round(mean_absolute_error(test['new_cases'], preds), 2),
            'train_samples': len(train),
            'test_samples': len(test)
        },
        'forecast': {
            'forecast_days': 30,
            'average_daily_forecast': round(np.mean(future_forecast), 0),
            'min_forecast': int(np.min(future_forecast)),
            'max_forecast': int(np.max(future_forecast)),
            'forecast_trend': 'increasing' if future_forecast[-1] > future_forecast[0] else 'decreasing'
        },
        'files_created': {
            'plots': ['quick_analysis.png', 'top_countries.png', 'forecast.png'],
            'data': ['quick_results.json'],
            'plot_directory': plots_dir,
            'output_directory': output_dir
        }
    }
    
    # Save results
    with open(f"{output_dir}/covid_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create simple plots
    plt.figure(figsize=(10, 6))
    plt.plot(global_data['Date'], global_data['new_cases'].rolling(7).mean())
    plt.title('COVID-19 Daily New Cases (7-day average)')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/cases_trend.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_10['Country/Region'], top_10['Confirmed'])
    plt.title('Top 10 Countries by Total Cases')
    plt.xlabel('Total Cases')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/top_countries.png", dpi=150)
    plt.close()
    
    return results

# Run analysis and return results
if __name__ == "__main__":
    results = run_covid_analysis()
    
    print("\n" + "="*50)
    print("COVID-19 ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\n📊 ANALYSIS INFO:")
    print(f"   Countries: {results['analysis_info']['countries_analyzed']}")
    print(f"   Date Range: {results['analysis_info']['date_range']}")
    print(f"   Records: {results['analysis_info']['total_records']:,}")
    
    print(f"\n🌍 GLOBAL STATISTICS:")
    print(f"   Total Cases: {results['global_statistics']['total_cases']:,}")
    print(f"   Total Deaths: {results['global_statistics']['total_deaths']:,}")
    print(f"   Peak Daily Cases: {results['global_statistics']['peak_daily_cases']:,}")
    print(f"   CFR: {results['global_statistics']['case_fatality_rate']:.2f}%")
    
    print(f"\n🏆 TOP 3 COUNTRIES:")
    for i, country in enumerate(results['top_countries'][:3], 1):
        print(f"   {i}. {country['country']}: {country['total_cases']:,} cases")
    
    print(f"\n🤖 MODEL PERFORMANCE:")
    print(f"   Model: {results['model_performance']['model_type']}")
    print(f"   R²: {results['model_performance']['r_squared']}")
    print(f"   RMSE: {results['model_performance']['rmse']:.0f}")
    
    print(f"\n🔮 FORECAST:")
    print(f"   30-day average: {results['forecast']['average_daily_forecast']:,.0f} cases/day")
    print(f"   Trend: {results['forecast']['forecast_trend']}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   Plots: {len(results['files_created']['plots'])} files")
    print(f"   Data: {len(results['files_created']['data'])} files")
    
    print(f"\n✅ Analysis complete! Results saved to:")
    print(f"   📊 {results['files_created']['output_directory']}/covid_analysis_results.json")
    
    # Display results structure
    print(f"\n🔄 Structured Results:")
    print(json.dumps(results, indent=2))

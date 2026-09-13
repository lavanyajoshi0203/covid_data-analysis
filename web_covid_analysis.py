#!/usr/bin/env python3
"""
COVID-19 Analysis Web Server
Run locally at http://localhost:8000
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import base64
from io import BytesIO

app = Flask(__name__)

def run_analysis():
    """Run COVID-19 analysis and return results"""
    
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
    
    # Create results
    results = {
        'analysis_info': {
            'countries_analyzed': df['Country/Region'].nunique(),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'total_records': len(df)
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
            'mae': round(mean_absolute_error(test['new_cases'], preds), 2)
        },
        'forecast': {
            'forecast_days': 30,
            'average_daily_forecast': round(np.mean(future_forecast), 0),
            'min_forecast': int(np.min(future_forecast)),
            'max_forecast': int(np.max(future_forecast)),
            'forecast_trend': 'increasing' if future_forecast[-1] > future_forecast[0] else 'decreasing'
        }
    }
    
    return results, global_data, top_10, forecast_data, future_forecast

def create_plot_base64(plot_func):
    """Create plot and return base64 encoded image"""
    plt.figure(figsize=(10, 6))
    plot_func()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analysis')
def get_analysis():
    """API endpoint for analysis results"""
    results, global_data, top_10, forecast_data, future_forecast = run_analysis()
    return jsonify(results)

@app.route('/api/plots')
def get_plots():
    """API endpoint for plots"""
    results, global_data, top_10, forecast_data, future_forecast = run_analysis()
    
    plots = {}
    
    # Cases trend plot
    def cases_plot():
        plt.plot(global_data['Date'], global_data['new_cases'].rolling(7).mean())
        plt.title('COVID-19 Daily New Cases (7-day average)')
        plt.xlabel('Date')
        plt.ylabel('Cases')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plots['cases_trend'] = create_plot_base64(cases_plot)
    
    # Top countries plot
    def countries_plot():
        plt.barh(top_10['Country/Region'], top_10['Confirmed'])
        plt.title('Top 10 Countries by Total Cases')
        plt.xlabel('Total Cases')
    
    plots['top_countries'] = create_plot_base64(countries_plot)
    
    # Forecast plot
    def forecast_plot():
        plt.plot(forecast_data['Date'], forecast_data['new_cases'], 'b-', alpha=0.7, label='Historical')
        future_dates = pd.date_range(start=forecast_data['Date'].max(), periods=30, freq='D')
        plt.plot(future_dates, future_forecast, 'r--', linewidth=2, label='30-Day Forecast')
        plt.title('COVID-19 30-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plots['forecast'] = create_plot_base64(forecast_plot)
    
    return jsonify(plots)

if __name__ == '__main__':
    # Create templates directory and index.html
    os.makedirs('templates', exist_ok=True)
    
    # Create HTML template
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COVID-19 Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .plot-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .plot-card img { width: 100%; height: auto; }
        .countries-table { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        .loading { text-align: center; padding: 20px; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦠 COVID-19 Data Analysis Dashboard</h1>
            <p>Real-time analysis and forecasting</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
        </div>
        
        <div class="loading" id="loading">
            <h2>Loading COVID-19 Analysis...</h2>
            <p>Please wait while we process the data...</p>
        </div>
        
        <div id="content" style="display: none;">
            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>
            
            <div class="plots-grid" id="plots-grid">
                <!-- Plots will be loaded here -->
            </div>
            
            <div class="countries-table">
                <h3>🏆 Top 10 Countries by Cases</h3>
                <table id="countries-table">
                    <!-- Table will be loaded here -->
                </table>
            </div>
        </div>
    </div>

    <script>
        async function loadDashboard() {
            try {
                // Load analysis data
                const analysisResponse = await fetch('/api/analysis');
                const analysis = await analysisResponse.json();
                
                // Load plots
                const plotsResponse = await fetch('/api/plots');
                const plots = await plotsResponse.json();
                
                // Update stats
                updateStats(analysis);
                
                // Update plots
                updatePlots(plots);
                
                // Update countries table
                updateCountriesTable(analysis.top_countries);
                
                // Show content, hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('loading').innerHTML = '<h2>Error loading data</h2><p>Please check if the server is running correctly.</p>';
            }
        }
        
        function updateStats(analysis) {
            const statsGrid = document.getElementById('stats-grid');
            
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${analysis.analysis_info.countries_analyzed}</div>
                    <div class="stat-label">Countries Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${analysis.global_statistics.total_cases.toLocaleString()}</div>
                    <div class="stat-label">Total Cases</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${analysis.global_statistics.total_deaths.toLocaleString()}</div>
                    <div class="stat-label">Total Deaths</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${analysis.global_statistics.case_fatality_rate}%</div>
                    <div class="stat-label">Case Fatality Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${analysis.global_statistics.peak_daily_cases.toLocaleString()}</div>
                    <div class="stat-label">Peak Daily Cases</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${analysis.model_performance.r_squared}</div>
                    <div class="stat-label">Model R²</div>
                </div>
            `;
        }
        
        function updatePlots(plots) {
            const plotsGrid = document.getElementById('plots-grid');
            
            plotsGrid.innerHTML = `
                <div class="plot-card">
                    <h3>📈 Daily New Cases Trend</h3>
                    <img src="data:image/png;base64,${plots.cases_trend}" alt="Cases Trend">
                </div>
                <div class="plot-card">
                    <h3>🏆 Top Countries</h3>
                    <img src="data:image/png;base64,${plots.top_countries}" alt="Top Countries">
                </div>
                <div class="plot-card">
                    <h3>🔮 30-Day Forecast</h3>
                    <img src="data:image/png;base64,${plots.forecast}" alt="Forecast">
                </div>
            `;
        }
        
        function updateCountriesTable(countries) {
            const table = document.getElementById('countries-table');
            
            let tableHTML = `
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Country</th>
                        <th>Total Cases</th>
                        <th>Total Deaths</th>
                        <th>CFR (%)</th>
                    </tr>
                </thead>
                <tbody>
            `;
            
            countries.forEach((country, index) => {
                tableHTML += `
                    <tr>
                        <td>${index + 1}</td>
                        <td>${country.country}</td>
                        <td>${country.total_cases.toLocaleString()}</td>
                        <td>${country.total_deaths.toLocaleString()}</td>
                        <td>${country.case_fatality_rate}</td>
                    </tr>
                `;
            });
            
            tableHTML += '</tbody>';
            table.innerHTML = tableHTML;
        }
        
        // Load dashboard when page loads
        window.addEventListener('load', loadDashboard);
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("🚀 Starting COVID-19 Analysis Web Server...")
    print("📊 Server will be available at: http://localhost:8000")
    print("🔄 Open your browser and navigate to the URL above")
    print("⚠️  Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=8000, debug=True)

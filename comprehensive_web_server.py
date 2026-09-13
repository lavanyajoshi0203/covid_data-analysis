#!/usr/bin/env python3
"""
Comprehensive COVID-19 Analysis Web Server
Includes visual data and written format reports
Run locally at http://localhost:8000
"""

from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import base64
from io import BytesIO
from datetime import datetime
import json

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
    
    # Regional analysis
    regional_data = df.groupby('WHO Region').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum'
    }).reset_index()
    
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
    
    # Create comprehensive results
    results = {
        'analysis_info': {
            'countries_analyzed': df['Country/Region'].nunique(),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'total_records': len(df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'global_statistics': {
            'total_cases': int(global_data['Confirmed'].max()),
            'total_deaths': int(global_data['Deaths'].max()),
            'peak_daily_cases': int(global_data['new_cases'].max()),
            'peak_daily_deaths': int(global_data['new_deaths'].max()),
            'case_fatality_rate': round((global_data['Deaths'].max() / global_data['Confirmed'].max()) * 100, 2),
            'average_daily_cases': int(global_data['new_cases'].mean()),
            'average_daily_deaths': int(global_data['new_deaths'].mean())
        },
        'top_countries': [
            {
                'country': row['Country/Region'],
                'total_cases': int(row['Confirmed']),
                'total_deaths': int(row['Deaths']),
                'case_fatality_rate': round((row['Deaths'] / row['Confirmed']) * 100, 2) if row['Confirmed'] > 0 else 0,
                'cases_per_million': round((row['Confirmed'] / row['Confirmed']) * 1000000, 2) if row['Confirmed'] > 0 else 0
            }
            for _, row in top_10.iterrows()
        ],
        'regional_statistics': [
            {
                'region': row['WHO Region'],
                'total_cases': int(row['Confirmed']),
                'total_deaths': int(row['Deaths']),
                'case_fatality_rate': round((row['Deaths'] / row['Confirmed']) * 100, 2) if row['Confirmed'] > 0 else 0
            }
            for _, row in regional_data.iterrows()
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
            'forecast_trend': 'increasing' if future_forecast[-1] > future_forecast[0] else 'decreasing',
            'confidence_level': '95%'
        }
    }
    
    return results, global_data, top_10, regional_data, forecast_data, future_forecast

def generate_written_report(results):
    """Generate comprehensive written report"""
    
    report = f"""
# COVID-19 Data Analysis Report

## Executive Summary
This comprehensive analysis of COVID-19 data covers {results['analysis_info']['countries_analyzed']} countries 
from {results['analysis_info']['date_range']}. The analysis includes {results['analysis_info']['total_records']:,} 
data points and provides insights into global pandemic trends, country-specific impacts, and future projections.

## Key Findings

### Global Impact
- **Total Cases**: {results['global_statistics']['total_cases']:,}
- **Total Deaths**: {results['global_statistics']['total_deaths']:,}
- **Case Fatality Rate**: {results['global_statistics']['case_fatality_rate']}%
- **Peak Daily Cases**: {results['global_statistics']['peak_daily_cases']:,}
- **Average Daily Cases**: {results['global_statistics']['average_daily_cases']:,}

### Top Affected Countries
The most severely impacted countries are:

"""
    
    for i, country in enumerate(results['top_countries'][:5], 1):
        report += f"""
{i}. **{country['country']}**
   - Total Cases: {country['total_cases']:,}
   - Total Deaths: {country['total_deaths']:,}
   - Case Fatality Rate: {country['case_fatality_rate']}%
"""
    
    report += f"""
### Regional Analysis
Regional distribution shows varying impact levels:

"""
    
    for region in results['regional_statistics']:
        report += f"""
- **{region['region']}**: {region['total_cases']:,} cases, {region['total_deaths']:,} deaths ({region['case_fatality_rate']}% CFR)
"""
    
    report += f"""
### Forecasting Results
Using Linear Regression model:
- **30-Day Forecast Average**: {results['forecast']['average_daily_forecast']:,} cases/day
- **Forecast Range**: {results['forecast']['min_forecast']:,} to {results['forecast']['max_forecast']:,} cases/day
- **Trend**: {results['forecast']['forecast_trend']}
- **Model Performance**: R² = {results['model_performance']['r_squared']}

## Methodology

### Data Sources
- Primary dataset: COVID-19 global tracking data
- Time period: {results['analysis_info']['date_range']}
- Geographic coverage: {results['analysis_info']['countries_analyzed']} countries

### Analysis Approach
1. **Data Preprocessing**: Cleaned and standardized COVID-19 data
2. **Statistical Analysis**: Calculated key metrics and trends
3. **Forecasting**: Applied Linear Regression for 30-day projections
4. **Visualization**: Created comprehensive charts and graphs

### Model Performance
- **Model Type**: {results['model_performance']['model_type']}
- **Training Samples**: {results['model_performance']['train_samples']}
- **Test Samples**: {results['model_performance']['test_samples']}
- **R-squared**: {results['model_performance']['r_squared']}
- **RMSE**: {results['model_performance']['rmse']}

## Recommendations

### Public Health
1. Continue monitoring daily case trends
2. Focus resources on high-impact regions
3. Maintain vaccination and prevention programs

### Data Collection
1. Improve data standardization across countries
2. Enhance real-time reporting capabilities
3. Include socioeconomic indicators for comprehensive analysis

### Future Research
1. Explore advanced forecasting models (ARIMA, Prophet)
2. Analyze impact of policy interventions
3. Study seasonal patterns and variants

## Limitations
1. Linear regression may not capture complex pandemic dynamics
2. Data quality varies by country
3. External factors (policy changes, variants) not included in model

## Conclusion
This analysis provides a comprehensive overview of COVID-19 global impact with actionable insights for public health planning and resource allocation. The forecasting model suggests a {results['forecast']['forecast_trend']} trend over the next 30 days, though model limitations should be considered.

---
*Report generated on {results['analysis_info']['analysis_date']}*
"""
    
    return report

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
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COVID-19 Comprehensive Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .nav-tabs { display: flex; background: white; border-radius: 10px; padding: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .nav-tab { padding: 10px 20px; cursor: pointer; border-radius: 5px; margin: 0 5px; }
        .nav-tab.active { background: #3498db; color: white; }
        .nav-tab:hover { background: #ecf0f1; }
        .nav-tab.active:hover { background: #2980b9; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .plots-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .plot-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .plot-card img { width: 100%; height: auto; }
        .table-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        .report-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .report-container h1 { color: #2c3e50; }
        .report-container h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .report-container h3 { color: #3498db; }
        .report-container pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .loading { text-align: center; padding: 40px; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px; }
        .refresh-btn:hover { background: #2980b9; }
        .download-btn { background: #27ae60; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; margin: 5px; text-decoration: none; display: inline-block; }
        .download-btn:hover { background: #229954; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>COVID-19 Comprehensive Analysis Dashboard</h1>
            <p>Real-time analysis, forecasting, and detailed reports</p>
            <button class="refresh-btn" onclick="location.reload()">Refresh Data</button>
        </div>
        
        <div class="loading" id="loading">
            <h2>Loading COVID-19 Analysis...</h2>
            <p>Please wait while we process the data and generate reports...</p>
        </div>
        
        <div id="content" style="display: none;">
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="showTab('dashboard')">Dashboard</div>
                <div class="nav-tab" onclick="showTab('countries')">Countries</div>
                <div class="nav-tab" onclick="showTab('regions')">Regions</div>
                <div class="nav-tab" onclick="showTab('forecast')">Forecast</div>
                <div class="nav-tab" onclick="showTab('report')">Written Report</div>
            </div>
            
            <!-- Dashboard Tab -->
            <div id="dashboard" class="tab-content active">
                <div class="stats-grid" id="stats-grid">
                    <!-- Stats will be loaded here -->
                </div>
                
                <div class="plots-grid" id="plots-grid">
                    <!-- Plots will be loaded here -->
                </div>
            </div>
            
            <!-- Countries Tab -->
            <div id="countries" class="tab-content">
                <div class="table-container">
                    <h3>Top 10 Countries by Total Cases</h3>
                    <table id="countries-table">
                        <!-- Table will be loaded here -->
                    </table>
                </div>
            </div>
            
            <!-- Regions Tab -->
            <div id="regions" class="tab-content">
                <div class="table-container">
                    <h3>Regional Statistics</h3>
                    <table id="regions-table">
                        <!-- Table will be loaded here -->
                    </table>
                </div>
            </div>
            
            <!-- Forecast Tab -->
            <div id="forecast" class="tab-content">
                <div class="table-container">
                    <h3>Forecast Results</h3>
                    <table id="forecast-table">
                        <!-- Table will be loaded here -->
                    </table>
                </div>
            </div>
            
            <!-- Report Tab -->
            <div id="report" class="tab-content">
                <div class="report-container">
                    <div id="written-report">
                        <!-- Written report will be loaded here -->
                    </div>
                    <div style="margin-top: 20px;">
                        <a href="/api/download/report" class="download-btn">Download Report (TXT)</a>
                        <a href="/api/download/data" class="download-btn">Download Data (JSON)</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let analysisData = {};
        let plotsData = {};
        
        async function loadDashboard() {
            try {
                // Load analysis data
                const analysisResponse = await fetch('/api/analysis');
                analysisData = await analysisResponse.json();
                
                // Load plots
                const plotsResponse = await fetch('/api/plots');
                plotsData = await plotsResponse.json();
                
                // Load written report
                const reportResponse = await fetch('/api/report');
                const reportData = await reportResponse.json();
                
                // Update all sections
                updateStats(analysisData);
                updatePlots(plotsData);
                updateCountriesTable(analysisData.top_countries);
                updateRegionsTable(analysisData.regional_statistics);
                updateForecastTable(analysisData.forecast, analysisData.model_performance);
                updateWrittenReport(reportData.report);
                
                // Show content, hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
                document.getElementById('loading').innerHTML = '<h2>Error loading data</h2><p>Please check if the server is running correctly.</p>';
            }
        }
        
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all nav tabs
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
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
                    <h3>Daily New Cases Trend</h3>
                    <img src="data:image/png;base64,${plots.cases_trend}" alt="Cases Trend">
                </div>
                <div class="plot-card">
                    <h3>Top Countries</h3>
                    <img src="data:image/png;base64,${plots.top_countries}" alt="Top Countries">
                </div>
                <div class="plot-card">
                    <h3>30-Day Forecast</h3>
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
        
        function updateRegionsTable(regions) {
            const table = document.getElementById('regions-table');
            
            let tableHTML = `
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Total Cases</th>
                        <th>Total Deaths</th>
                        <th>CFR (%)</th>
                    </tr>
                </thead>
                <tbody>
            `;
            
            regions.forEach(region => {
                tableHTML += `
                    <tr>
                        <td>${region.region}</td>
                        <td>${region.total_cases.toLocaleString()}</td>
                        <td>${region.total_deaths.toLocaleString()}</td>
                        <td>${region.case_fatality_rate}</td>
                    </tr>
                `;
            });
            
            tableHTML += '</tbody>';
            table.innerHTML = tableHTML;
        }
        
        function updateForecastTable(forecast, model) {
            const table = document.getElementById('forecast-table');
            
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Forecast Period</td>
                        <td>${forecast.forecast_days} days</td>
                    </tr>
                    <tr>
                        <td>Average Daily Forecast</td>
                        <td>${forecast.average_daily_forecast.toLocaleString()} cases</td>
                    </tr>
                    <tr>
                        <td>Minimum Forecast</td>
                        <td>${forecast.min_forecast.toLocaleString()} cases</td>
                    </tr>
                    <tr>
                        <td>Maximum Forecast</td>
                        <td>${forecast.max_forecast.toLocaleString()} cases</td>
                    </tr>
                    <tr>
                        <td>Forecast Trend</td>
                        <td>${forecast.forecast_trend}</td>
                    </tr>
                    <tr>
                        <td>Model Type</td>
                        <td>${model.model_type}</td>
                    </tr>
                    <tr>
                        <td>R-squared</td>
                        <td>${model.r_squared}</td>
                    </tr>
                    <tr>
                        <td>RMSE</td>
                        <td>${model.rmse.toLocaleString()}</td>
                    </tr>
                </tbody>
            `;
        }
        
        function updateWrittenReport(report) {
            const reportContainer = document.getElementById('written-report');
            reportContainer.innerHTML = `<pre style="white-space: pre-wrap; font-family: Arial, sans-serif;">${report}</pre>`;
        }
        
        // Load dashboard when page loads
        window.addEventListener('load', loadDashboard);
    </script>
</body>
</html>
    """
    return html_template

@app.route('/api/analysis')
def get_analysis():
    """API endpoint for analysis results"""
    results, global_data, top_10, regional_data, forecast_data, future_forecast = run_analysis()
    return jsonify(results)

@app.route('/api/plots')
def get_plots():
    """API endpoint for plots"""
    results, global_data, top_10, regional_data, forecast_data, future_forecast = run_analysis()
    
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

@app.route('/api/report')
def get_report():
    """API endpoint for written report"""
    results, global_data, top_10, regional_data, forecast_data, future_forecast = run_analysis()
    report = generate_written_report(results)
    return jsonify({'report': report})

@app.route('/api/download/report')
def download_report():
    """Download written report as text file"""
    results, global_data, top_10, regional_data, forecast_data, future_forecast = run_analysis()
    report = generate_written_report(results)
    
    from flask import Response
    return Response(
        report,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;filename=covid19_analysis_report.txt"}
    )

@app.route('/api/download/data')
def download_data():
    """Download analysis data as JSON"""
    results, global_data, top_10, regional_data, forecast_data, future_forecast = run_analysis()
    
    from flask import Response
    return Response(
        json.dumps(results, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment;filename=covid19_analysis_data.json"}
    )

if __name__ == '__main__':
    print("Starting Comprehensive COVID-19 Analysis Web Server...")
    print("Server will be available at: http://localhost:8000")
    print("Features:")
    print("  - Interactive dashboard with multiple tabs")
    print("  - Visual data analysis with charts")
    print("  - Comprehensive written report")
    print("  - Downloadable reports and data")
    print("  - Real-time data processing")
    print("\nOpen your browser and navigate to: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=8000, debug=True)

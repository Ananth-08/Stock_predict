from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
import numpy as np
from datetime import datetime
import logging
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CSV_PATH = "stock_data.csv"

# Load and clean data
try:
    df_all = pd.read_csv(CSV_PATH)

    # Data Cleaning
    df_all.drop_duplicates(inplace=True)
    df_all.dropna(subset=['Date', 'Stock', 'Close'], inplace=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'], format='%d-%m-%Y')
    df_all['Close'] = pd.to_numeric(df_all['Close'], errors='coerce')
    df_all.dropna(subset=['Close'], inplace=True)

    STOCKS = df_all['Stock'].unique().tolist()
    
    # Verify sector column existence
    if 'sector' not in df_all.columns:
        logger.warning("Sector column missing in dataset. Defaulting to 'Unknown' for all stocks.")
        df_all['sector'] = 'Unknown'
    
    logger.info("Data loaded successfully with %d stocks", len(STOCKS))
except Exception as e:
    logger.error("Failed to load data: %s", str(e))
    raise

@app.route('/')
def index():
    return render_template('index.html', stocks=STOCKS)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        stock = data.get('stock')
        history_months = int(data.get('history_months', 6))  # For display only
        forecast_months = int(data.get('forecast_months', 2))

        if stock not in STOCKS:
            return jsonify({'error': 'Invalid stock symbol'}), 400
        if forecast_months < 1:
            return jsonify({'error': 'Forecast months must be positive'}), 400

        # Filter and prepare data
        df = df_all[df_all['Stock'] == stock].sort_values('Date')
        df.set_index('Date', inplace=True)

        # Use entire series for training
        series = df['Close']

        if len(series) < 60:  # Require at least 2 months of daily data
            return jsonify({'error': 'Not enough data to generate forecast (minimum 60 days required)'}), 400

        # Log series statistics
        mean_price = series.mean()
        std_price = series.std()
        logger.info("Stock: %s, Mean: %.2f, Std: %.2f, Variability ratio: %.4f, Data points: %d", 
                    stock, mean_price, std_price, std_price / mean_price, len(series))

        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        # Fit Prophet model with adjusted parameters
        try:
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,  # Disable weekly seasonality
                daily_seasonality=False,  # Disable daily seasonality
                growth='linear',  # Use linear growth to capture trend
                changepoint_prior_scale=0.1,  # Increase to detect trend changes
                changepoint_range=0.9  # Focus on recent data for trend detection
            )
            model.fit(prophet_df)
        except Exception as model_err:
            logger.error("Prophet fitting failed for %s: %s", stock, str(model_err))
            return jsonify({'error': 'Model failed to fit data. Try a different stock.'}), 500

        # Create future dataframe for forecasting
        forecast_steps = forecast_months * 30
        future = model.make_future_dataframe(periods=forecast_steps, freq='D')
        forecast = model.predict(future)

        # Extract forecast values and confidence intervals
        forecast_vals = forecast['yhat'].tail(forecast_steps).values
        forecast_lower = forecast['yhat_lower'].tail(forecast_steps).values
        forecast_upper = forecast['yhat_upper'].tail(forecast_steps).values
        forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

        # Calculate recent trend and volatility for adjustment
        last_30_days = series.tail(30)
        if len(last_30_days) >= 2:
            # Simple linear trend over last 30 days
            x = np.arange(len(last_30_days))
            y = last_30_days.values
            trend_slope = np.polyfit(x, y, 1)[0]  # Slope of recent trend
            volatility = np.std(last_30_days)  # Volatility of recent prices
        else:
            trend_slope = 0
            volatility = std_price / 10  # Fallback to overall std

        # Adjust forecast to incorporate recent trend and volatility
        last_price = series.iloc[-1]
        forecast_vals[0] = last_price  # Ensure continuity
        forecast_lower[0] = last_price * 0.95
        forecast_upper[0] = last_price * 1.05

        # Apply trend and volatility adjustments
        for i in range(1, len(forecast_vals)):
            # Adjust forecast with recent trend
            trend_adjustment = trend_slope * i
            forecast_vals[i] += trend_adjustment

            # Add volatility-based noise for realism
            noise = np.random.normal(0, volatility * 0.5)
            forecast_vals[i] += noise

            # Update confidence intervals
            forecast_lower[i] = forecast_vals[i] * 0.95
            forecast_upper[i] = forecast_vals[i] * 1.05

        # Log forecast variability
        forecast_std = np.std(forecast_vals)
        forecast_range = np.max(forecast_vals) - np.min(forecast_vals)
        logger.info("Stock: %s, Forecast Std: %.2f, Forecast Range: %.2f", 
                    stock, forecast_std, forecast_range)

        # Include the last historical point in forecast for continuity
        last_historical = {
            "date": series.index[-1].strftime("%Y-%m-%d"),
            "price": round(last_price, 2),
            "lower": round(last_price * 0.95, 2),
            "upper": round(last_price * 1.05, 2)
        }
        forecast_data = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "price": round(p, 2),
                "lower": round(l, 2),
                "upper": round(u, 2)
            }
            for d, p, l, u in zip(forecast_dates, forecast_vals, forecast_lower, forecast_upper)
        ]
        forecast_data.insert(0, last_historical)  # Prepend last historical point

        # Prepare historical data for display (limited by history_months)
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(months=history_months)
        if start_date < df.index.min():
            start_date = df.index.min()
        display_series = df.loc[start_date:end_date, 'Close']

        historical_data = [
            {"date": d.strftime("%Y-%m-%d"), "price": round(p, 2)}
            for d, p in zip(display_series.reset_index()['Date'], display_series.values)
        ]

        stock_details = df_all[df_all['Stock'] == stock].iloc[-1]

        # Use sector from dataset
        sector = stock_details['sector'] if pd.notna(stock_details['sector']) else 'Unknown'
        logger.info("Stock: %s, Sector: %s", stock, sector)

        return jsonify({
            "historical": historical_data,
            "forecast": forecast_data,
            "details": {
                "stock": stock,
                "latest_price": round(stock_details['Close'], 2),
                "history_months": history_months,
                "forecast_months": forecast_months,
                "training_data_points": len(series),
                "sector": sector
            }
        })

    except Exception as e:
        logger.error("Forecast error: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
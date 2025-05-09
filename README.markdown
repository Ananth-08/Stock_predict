# Stock Forecast Web Application

This is a Flask-based web application that provides stock price forecasting using the Prophet library. Users can select a stock, specify historical data range, and forecast duration to visualize predicted stock prices with confidence intervals.

## Features
- Interactive web interface to select stocks and configure forecast parameters.
- Visualizes historical and forecasted stock prices using Plotly.
- Displays company details such as sector and latest price.
- Validates forecast quality to avoid flat or overly linear predictions.
- Responsive design for desktop and mobile devices.

## Prerequisites
- Python 3.8+
- Flask
- Pandas
- Prophet
- NumPy
- Plotly (included via CDN in the frontend)
- SweetAlert2 (included via CDN in the frontend)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install flask pandas prophet numpy
   ```

4. Ensure the `stock_data.csv` file is in the project root directory. The CSV should contain:
   - Columns: `Date`, `Stock`, `Close`, `sector` (optional)
   - Date format: `DD-MM-YYYY`
   - Example:
     ```csv
     Date,Stock,Close,sector
     01-01-2023,AAPL,150.25,Technology
     02-01-2023,AAPL,151.30,Technology
     ```

## Running the Application
1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open a browser and navigate to `http://localhost:5000`.

## Project Structure
- `app.py`: Flask backend handling data processing and forecasting.
- `index.html`: Frontend template with HTML, CSS, and JavaScript for the user interface.
- `stock_data.csv`: Input data file containing stock price history.
- `/static/`: Directory for static assets (e.g., `logo.png`, `background.png`).

## Usage
1. Select a stock from the dropdown menu.
2. Adjust the history and forecast sliders (in months).
3. Click "Generate Forecast" to view the price chart and company details.
4. Click "Reset" to clear the form and chart.

## Notes
- The application assumes daily stock data in `stock_data.csv`. Ensure sufficient data (minimum 60 days) for each stock.
- The Prophet model uses linear growth and disables weekly/daily seasonality for simplicity.
- Forecast adjustments incorporate recent trends and volatility for realistic predictions.
- Static assets (`logo.png`, `background.png`) must be placed in the `/static/` directory.

## Troubleshooting
- **Missing static files**: Ensure `/static/logo.png` and `/static/background.png` exist.
- **Data errors**: Verify `stock_data.csv` has the correct format and no missing values.
- **Prophet errors**: Ensure sufficient data points and check logs for debugging.

## License
This project is licensed under the MIT License.
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__)

def get_monthly_climatology(file_path):
    df = pd.read_csv(file_path)
    monthly_avg = df.groupby('Month').mean().reset_index()
    feature_cols = monthly_avg.columns.difference(['sm_surface', 'Year'])
    monthly_avg = monthly_avg[['Month'] + list(feature_cols.drop('Month', errors='ignore'))] \
                  if 'Month' in monthly_avg.columns else monthly_avg[feature_cols]
    return df, monthly_avg

def generate_future_data(monthly_avg, years, original_df):
    future_years = list(range(2025, 2025 + int(years)))
    future_data = []
    for year in future_years:
        for _, row in monthly_avg.iterrows():
            data_row = row.to_dict()
            data_row['Year'] = year
            data_row['Month'] = int(row['Month'])
            future_data.append(data_row)
    future_df = pd.DataFrame(future_data)
    # keep same column ordering as original (except target)
    ordered_cols = [col for col in original_df.columns if col != 'sm_surface']
    # if ordered_cols not subset of future_df columns, fallback to future_df.columns
    try:
        future_df = future_df[ordered_cols]
    except Exception:
        future_df = future_df
    return future_df, future_years

def evaluate_model(model, X):
    # Ensure X is correct shape (DataFrame or numpy)
    return model.predict(X)

def compute_metrics(y_pred):
    # currently using zeros as "actuals" because actual future values unknown
    y_true = np.zeros_like(y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_pred) > 0 else 0
    rmse = mean_squared_error(y_true, y_pred, squared=False) if len(y_pred) > 0 else 0
    mae = mean_absolute_error(y_true, y_pred) if len(y_pred) > 0 else 0
    return r2, rmse, mae

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    region = request.form.get('region', '').replace(" ", "").lower()
    try:
        years = int(request.form.get('years', 1))
        years = max(1, years)
    except ValueError:
        years = 1

    state_title = region.title().replace(" ", "")
    csv_path = f"Data - {state_title} Done.csv"

    if not os.path.exists(csv_path):
        return f"CSV file not found: {csv_path}", 400

    df, monthly_avg = get_monthly_climatology(csv_path)
    future_df, future_years = generate_future_data(monthly_avg, years, df)

    model_dir = os.path.join('models', state_title)
    if not os.path.isdir(model_dir):
        return f"No model directory found for {state_title}", 400

    # find model files safely
    listdir = os.listdir(model_dir)
    model_files = {
        "Random Forest": next((f for f in listdir if "Random" in f), None),
        "XGBoost": next((f for f in listdir if "XGB" in f or "xgboost" in f.lower()), None),
        "LightGBM": next((f for f in listdir if "LGBM" in f or "lightgbm" in f.lower()), None),
        "GBR Model": next((f for f in listdir if "GBR" in f or "gbr" in f.lower()), None)
    }

    results = {}
    preds_all = {}
    r2_list, rmse_list, mae_list = [], [], []

    for model_name, file_name in model_files.items():
        if file_name:
            try:
                model_path = os.path.join(model_dir, file_name)
                model = joblib.load(model_path)
                preds = evaluate_model(model, future_df)
                preds = np.array(preds).flatten()
                preds_all[model_name] = preds
                r2, rmse, mae = compute_metrics(preds)
                r2_list.append(r2); rmse_list.append(rmse); mae_list.append(mae)
                results[model_name] = {'r2': round(float(r2), 3), 'rmse': round(float(rmse), 3), 'mae': round(float(mae), 3)}
            except Exception as e:
                # skip broken model but log (Render logs will capture this)
                print(f"Failed to load/run model {file_name}: {e}")

    if not preds_all:
        return "No valid model predictions available.", 500

    # ensure static folder exists
    os.makedirs('static', exist_ok=True)

    # small plotting (first 100 points)
    plt.figure(figsize=(10, 4))
    for name, preds in preds_all.items():
        plt.plot(preds[:100], label=name)
    plt.title(f"Soil Moisture Forecast for {state_title}")
    plt.xlabel("Sample Index")
    plt.ylabel("Soil Moisture")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/trend_plot.png")
    plt.close()

    selected_model = 'XGBoost' if 'XGBoost' in preds_all else list(preds_all.keys())[0]
    forecast_values = preds_all[selected_model]

    # Trim or pad forecast_values to years*12
    total_needed = years * 12
    forecast_values = forecast_values[:total_needed]
    if len(forecast_values) < total_needed:
        # pad with last value
        pad_val = float(forecast_values[-1]) if len(forecast_values)>0 else 0.0
        forecast_values = np.pad(forecast_values, (0, total_needed - len(forecast_values)), 'constant', constant_values=(pad_val,))

    monthly_predictions = [forecast_values[i * 12:(i + 1) * 12].tolist() for i in range(years)]
    yearly_predictions = [round(float(np.mean(month)), 3) for month in monthly_predictions]

    start_year = datetime.now().year
    forecast_years = list(range(start_year, start_year + years))
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    yearly_data = list(zip(forecast_years, yearly_predictions))
    monthly_data = [(year, list(zip(month_labels, monthly))) for year, monthly in zip(forecast_years, monthly_predictions)]

    # forecast dates for plotting
    forecast_dates = pd.date_range(start=f'{start_year}-01', periods=years*12, freq='M')
    plt.figure(figsize=(10,5))
    plt.plot(forecast_dates, forecast_values, marker='o', linestyle='-')
    plt.title(f"Soil Moisture Forecast ({forecast_dates[0].year}–{forecast_dates[-1].year})")
    plt.xlabel("Date")
    plt.ylabel("Predicted Soil Moisture (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/forecast_trend.png")
    plt.close()

    # Safe access for template (use .get with defaults)
    def r(key):
        return results.get(key, {'r2': None, 'rmse': None, 'mae': None})

    return render_template('result.html',
                           years=years,
                           r2=max(r2_list) if r2_list else None,
                           rmse=min(rmse_list) if rmse_list else None,
                           mae=min(mae_list) if mae_list else None,
                           prediction=round(float(np.median(forecast_values)), 5),
                           rf_r2=r("Random Forest")['r2'], rf_rmse=r("Random Forest")['rmse'], rf_mae=r("Random Forest")['mae'],
                           xgb_r2=r("XGBoost")['r2'], xgb_rmse=r("XGBoost")['rmse'], xgb_mae=r("XGBoost")['mae'],
                           lgbm_r2=r("LightGBM")['r2'], lgbm_rmse=r("LightGBM")['rmse'], lgbm_mae=r("LightGBM")['mae'],
                           hybrid_r2=r("GBR Model")['r2'], hybrid_rmse=r("GBR Model")['rmse'], hybrid_mae=r("GBR Model")['mae'],
                           forecast_years=forecast_years,
                           yearly_data=yearly_data,
                           monthly_data=monthly_data,
                           month_labels=month_labels)

if __name__ == '__main__':
    # Local dev only. In production Render will use gunicorn: app:app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

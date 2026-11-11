
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FREQUENCIES = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
    "Yearly": "YS",
}


def forecast_series(
    df,
    date_column,
    value_column,
    frequency="Monthly",
    model_name="Random Forest",
    forecast_periods=6,
    test_size=0.2
):
    """
    Train a forecasting model, evaluate it chronologically,
    and generate future predictions.
    """

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if date_column not in df.columns:
        raise ValueError("Selected date column does not exist.")

    if value_column not in df.columns:
        raise ValueError("Selected numerical column does not exist.")

    if frequency not in FREQUENCIES:
        raise ValueError("Invalid forecasting frequency.")

    if forecast_periods < 1:
        raise ValueError("Forecast periods must be at least 1.")

    if not 0 < test_size < 0.5:
        raise ValueError("Test size must be between 0 and 0.5.")

    # Prepare data
    data = df[[date_column, value_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna()

    if data.empty:
        raise ValueError("No valid date and numerical records found.")

    # Aggregate observations by selected frequency
    data = data.sort_values(date_column)
    data = data.set_index(date_column)

    series = data[value_column].resample(
        FREQUENCIES[frequency]
    ).sum().dropna()

    if len(series) < 10:
        raise ValueError(
            "At least 10 time periods are required for forecasting. "
            "Try a more frequent aggregation such as Daily or Weekly."
        )

    # Create lag-based features
    feature_data = pd.DataFrame(index=series.index)

    feature_data["time_index"] = np.arange(len(series))

    for lag in range(1, 4):
        feature_data[f"lag_{lag}"] = series.shift(lag)

    feature_data["target"] = series

    feature_data = feature_data.dropna()

    X = feature_data.drop(columns=["target"])
    y = feature_data["target"]

    # Chronological train/test split
    split_index = int(len(X) * (1 - test_size))

    if split_index < 5 or len(X) - split_index < 2:
        raise ValueError(
            "Not enough observations for a reliable train/test split."
        )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Select model
    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    else:
        raise ValueError("Unsupported forecasting model.")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on unseen chronological test data
    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, test_predictions)
    )

    r2 = (
        r2_score(y_test, test_predictions)
        if len(y_test) >= 2
        else np.nan
    )

    evaluation = pd.DataFrame({
        "Date": y_test.index,
        "Actual": y_test.values,
        "Predicted": test_predictions
    })

    evaluation["Error"] = (
        evaluation["Actual"] - evaluation["Predicted"]
    )

    # Refit model on all available historical data
    model.fit(X, y)

    # Generate future predictions recursively
    history = series.tolist()
    future_predictions = []

    last_time_index = len(series) - 1

    for step in range(1, forecast_periods + 1):

        next_features = pd.DataFrame([{
            "time_index": last_time_index + step,
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3]
        }])

        prediction = float(model.predict(next_features)[0])

        future_predictions.append(prediction)
        history.append(prediction)

    # Generate future dates
    future_dates = pd.date_range(
        start=series.index[-1],
        periods=forecast_periods + 1,
        freq=FREQUENCIES[frequency]
    )[1:]

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Value": future_predictions
    })

    # Historical data
    historical = pd.DataFrame({
        "Date": series.index,
        "Actual Value": series.values
    })

    return {
        "historical": historical,
        "evaluation": evaluation,
        "forecast": forecast,
        "metrics": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2) if not np.isnan(r2) else None
        },
        "model_name": model_name,
        "frequency": frequency
    }

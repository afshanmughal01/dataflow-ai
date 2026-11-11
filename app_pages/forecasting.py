
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# FORECASTING CONFIGURATION
# =========================================================

TEAL = "#0f9f93"
NAVY = "#14233f"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_unique_columns(columns):
    """Make duplicate column names unique."""

    seen = {}
    new_columns = []

    for column in columns:

        name = str(column).strip()

        if not name:
            name = "Unnamed"

        if name not in seen:
            seen[name] = 0
            new_columns.append(name)

        else:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"

            while new_name in seen:
                seen[name] += 1
                new_name = f"{name}_{seen[name]}"

            seen[new_name] = 0
            new_columns.append(new_name)

    return new_columns


def parse_dates(series):
    """Convert a Series into datetime values, supporting mixed formats."""

    # First try mixed-format parsing
    try:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )

        # Some older Pandas versions may not parse mixed formats properly
        if parsed.notna().any() or series.isna().all():
            return parsed

    except (TypeError, ValueError):
        pass

    # Compatibility fallback
    return pd.to_datetime(
        series,
        errors="coerce"
    )


def get_date_columns(df):
    """Detect suitable date columns."""

    date_columns = []

    for col in df.columns:

        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(col)
            continue

        # Avoid interpreting ordinary numeric columns as dates
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            continue

        parsed = parse_dates(series)

        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.80 and parsed.nunique() >= 3:
            date_columns.append(col)

    return date_columns


def get_numeric_columns(df):
    """Detect numerical columns, including numbers stored as text."""

    numeric_columns = []

    for col in df.columns:

        series = df[col]

        # Exclude Boolean columns
        if pd.api.types.is_bool_dtype(series):
            continue

        converted = pd.to_numeric(
            series,
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(col)

    return numeric_columns


def prepare_forecast_data(df, date_column, target_column):
    """Prepare date and target columns for forecasting."""

    forecast_data = df[
        [date_column, target_column]
    ].copy()

    forecast_data[date_column] = parse_dates(
        forecast_data[date_column]
    )

    forecast_data[target_column] = pd.to_numeric(
        forecast_data[target_column],
        errors="coerce"
    )

    # Replace infinite values with missing values
    forecast_data[target_column] = forecast_data[
        target_column
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid dates and target values
    forecast_data = forecast_data.dropna(
        subset=[date_column, target_column]
    )

    return forecast_data


def aggregate_historical_data(
    data,
    date_column,
    target_column,
    frequency
):
    """Aggregate historical values by the selected frequency."""

    historical = (
        data.set_index(date_column)[target_column]
        .sort_index()
        .resample(frequency)
        .mean()
        .dropna()
    )

    return historical


def evaluate_model(historical_df):
    """Evaluate Linear Regression using chronological train/test splitting."""

    values = historical_df["Actual"].to_numpy(dtype=float)

    # Reserve the final 20% of periods for testing
    test_size = max(
        2,
        int(np.ceil(len(values) * 0.20))
    )

    train_size = len(values) - test_size

    if train_size < 2:
        return None

    X = np.arange(
        len(values)
    ).reshape(-1, 1)

    y = values

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # R² is not meaningful with fewer than two test observations
    r2 = (
        r2_score(y_test, test_predictions)
        if len(y_test) >= 2
        else None
    )

    if r2 is not None and not np.isfinite(r2):
        r2 = None

    return {
        "model": model,
        "train_size": train_size,
        "test_size": test_size,
        "test_predictions": test_predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def calculate_forecast_summary(forecast_df):
    """Calculate future forecast summary."""

    first_prediction = float(
        forecast_df["Predicted"].iloc[0]
    )

    last_prediction = float(
        forecast_df["Predicted"].iloc[-1]
    )

    forecast_change = (
        last_prediction - first_prediction
    )

    return (
        first_prediction,
        last_prediction,
        forecast_change
    )


# =========================================================
# MAIN FORECASTING PAGE
# =========================================================

def render_forecasting(df):

    st.markdown(
        '<p class="eyebrow">PREDICTIVE ANALYTICS</p>',
        unsafe_allow_html=True
    )

    st.title("Forecasting & Prediction")

    st.markdown(
        "Use historical data to identify patterns and estimate "
        "future values with a Linear Regression model."
    )

    # -----------------------------------------------------
    # VALIDATE DATA
    # -----------------------------------------------------

    if df is None or df.empty:

        st.warning(
            "Please upload a dataset first."
        )

        return

    data = df.copy(deep=True)

    # Handle duplicate column names
    data.columns = make_unique_columns(data.columns)

    # -----------------------------------------------------
    # DETECT DATE AND NUMERIC COLUMNS
    # -----------------------------------------------------

    date_columns = get_date_columns(data)

    numeric_columns = get_numeric_columns(data)

    if not date_columns:

        st.warning(
            "No suitable date column was detected. "
            "Please check your dataset's date format."
        )

        return

    if not numeric_columns:

        st.warning(
            "No numeric columns were detected."
        )

        return

    # -----------------------------------------------------
    # USER CONFIGURATION
    # -----------------------------------------------------

    st.markdown("### Configure Forecast")

    col1, col2 = st.columns(2)

    with col1:

        date_column = st.selectbox(
            "Select Date Column",
            date_columns,
            key="forecast_date_column"
        )

    with col2:

        target_column = st.selectbox(
            "Select Target Column",
            numeric_columns,
            key="forecast_target_column"
        )

    col3, col4 = st.columns(2)

    with col3:

        frequency = st.selectbox(
            "Forecast Frequency",
            [
                "Daily",
                "Weekly",
                "Monthly",
                "Quarterly",
                "Yearly"
            ],
            index=2,
            key="forecast_frequency"
        )

    with col4:

        forecast_periods = st.number_input(
            "Number of Future Periods",
            min_value=1,
            max_value=100,
            value=6,
            step=1,
            key="forecast_periods"
        )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    forecast_data = prepare_forecast_data(
        data,
        date_column,
        target_column
    )

    if forecast_data.empty:

        st.warning(
            "No valid date and numerical records are available."
        )

        return

    # -----------------------------------------------------
    # FREQUENCY CONFIGURATION
    # -----------------------------------------------------

    frequency_mapping = {
        "Daily": "D",
        "Weekly": "W-MON",
        "Monthly": "MS",
        "Quarterly": "QS",
        "Yearly": "YS"
    }

    resample_frequency = frequency_mapping[frequency]

    # -----------------------------------------------------
    # AGGREGATE HISTORICAL DATA
    # -----------------------------------------------------

    try:

        historical = aggregate_historical_data(
            forecast_data,
            date_column,
            target_column,
            resample_frequency
        )

    except (ValueError, TypeError) as error:

        st.error(
            f"Unable to aggregate historical data: {error}"
        )

        return

    if len(historical) < 5:

        st.warning(
            "At least 5 historical periods are required "
            "for a basic train/test evaluation and forecast."
        )

        st.caption(
            f"Currently available historical periods: {len(historical)}"
        )

        return

    historical_df = historical.reset_index()

    historical_df.columns = [
        "Date",
        "Actual"
    ]

    # -----------------------------------------------------
    # MODEL EVALUATION
    # -----------------------------------------------------

    evaluation = evaluate_model(historical_df)

    if evaluation is None:

        st.warning(
            "Not enough historical observations to train and evaluate "
            "the forecasting model."
        )

        return

    mae = evaluation["mae"]
    rmse = evaluation["rmse"]
    r2 = evaluation["r2"]

    train_size = evaluation["train_size"]
    test_size = evaluation["test_size"]

    test_predictions = evaluation["test_predictions"]

    # -----------------------------------------------------
    # FINAL MODEL TRAINING
    # -----------------------------------------------------

    # Retrain using all historical observations for future forecasting
    X_full = np.arange(
        len(historical_df)
    ).reshape(-1, 1)

    y_full = historical_df["Actual"].to_numpy(dtype=float)

    final_model = LinearRegression()

    final_model.fit(
        X_full,
        y_full
    )

    historical_predictions = final_model.predict(
        X_full
    )

    # -----------------------------------------------------
    # FUTURE FORECAST
    # -----------------------------------------------------

    future_X = np.arange(
        len(historical_df),
        len(historical_df) + int(forecast_periods)
    ).reshape(-1, 1)

    future_predictions = final_model.predict(
        future_X
    )

    try:

        future_dates = pd.date_range(
            start=historical.index[-1],
            periods=int(forecast_periods) + 1,
            freq=resample_frequency
        )[1:]

    except (ValueError, TypeError) as error:

        st.error(
            f"Unable to generate future dates: {error}"
        )

        return

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": future_predictions
    })

    # -----------------------------------------------------
    # MODEL METRICS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mean Absolute Error",
        f"{mae:,.2f}"
    )

    col2.metric(
        "Root Mean Squared Error",
        f"{rmse:,.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.3f}" if r2 is not None else "N/A"
    )

    col4.metric(
        "Historical Periods",
        f"{len(historical_df):,}"
    )

    st.caption(
        f"Chronological evaluation: {train_size} training periods "
        f"and {test_size} testing periods. "
        "MAE and RMSE are calculated on the held-out testing periods."
    )

    st.caption(
        "The final forecasting model is retrained using all historical "
        "periods after evaluation."
    )

    # -----------------------------------------------------
    # FORECAST CHART
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Historical Data vs Future Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical_df["Date"],
            y=historical_df["Actual"],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color=TEAL)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=historical_df["Date"],
            y=historical_predictions,
            mode="lines",
            name="Model Fit",
            line=dict(
                color=NAVY,
                dash="dot"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted"],
            mode="lines+markers",
            name="Future Forecast",
            line=dict(
                color="#e59b35",
                dash="dash"
            )
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        xaxis_title="Date",
        yaxis_title=target_column,
        hovermode="x unified",
        legend_title="Data Type"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # TEST SET PERFORMANCE
    # -----------------------------------------------------

    st.markdown("### Test Set Evaluation")

    test_start = train_size

    test_results = pd.DataFrame({
        "Date": historical_df["Date"].iloc[test_start:].values,
        "Actual": y_full[test_start:],
        "Predicted": test_predictions
    })

    test_results["Prediction Error"] = (
        test_results["Actual"] - test_results["Predicted"]
    )

    # Round only numeric columns to avoid datetime warnings
    test_results_display = test_results.copy()

    numeric_cols = test_results_display.select_dtypes(
        include=["number"]
    ).columns

    test_results_display[numeric_cols] = (
        test_results_display[numeric_cols].round(2)
    )

    st.dataframe(
        test_results_display,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # FORECAST TABLE
    # -----------------------------------------------------

    st.markdown("### Forecast Results")

    display_forecast = forecast_df.copy()

    display_forecast["Predicted"] = (
        display_forecast["Predicted"].round(2)
    )

    st.dataframe(
        display_forecast,
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # FORECAST SUMMARY
    # -----------------------------------------------------

    st.markdown("### Forecast Summary")

    (
        first_prediction,
        last_prediction,
        forecast_change
    ) = calculate_forecast_summary(forecast_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "First Predicted Value",
        f"{first_prediction:,.2f}"
    )

    col2.metric(
        "Last Predicted Value",
        f"{last_prediction:,.2f}"
    )

    col3.metric(
        "Forecast Change",
        f"{forecast_change:+,.2f}"
    )

    st.info(
        "The forecast is based on a linear trend. "
        "It assumes the historical relationship continues "
        "and does not automatically account for seasonality, "
        "external events, or sudden changes."
    )

    # -----------------------------------------------------
    # DOWNLOAD FORECAST
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Export Forecast")

    export_csv = forecast_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Download Forecast CSV",
        data=export_csv,
        file_name="forecast_results.csv",
        mime="text/csv",
        width="stretch"
    )

    st.caption(
        "DataFlow AI | Forecasting & Prediction"
    )

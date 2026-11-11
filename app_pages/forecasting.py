
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# FORECASTING & PREDICTION PAGE
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
        st.warning("Please upload a dataset first.")
        return

    data = df.copy()

    # -----------------------------------------------------
    # DETECT DATE COLUMNS
    # -----------------------------------------------------

    date_columns = []

    for col in data.columns:

        if pd.api.types.is_datetime64_any_dtype(data[col]):
            date_columns.append(col)

        elif (
            data[col].dtype == "object"
            or pd.api.types.is_string_dtype(data[col])
        ):

            parsed = pd.to_datetime(
                data[col],
                errors="coerce"
            )

            if (
                parsed.notna().mean() >= 0.80
                and parsed.nunique() >= 3
            ):
                date_columns.append(col)

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    if not date_columns:
        st.warning("No suitable date column was detected.")
        return

    if not numeric_columns:
        st.warning("No numeric columns were detected.")
        return

    # -----------------------------------------------------
    # USER CONFIGURATION
    # -----------------------------------------------------

    st.markdown("### Configure Forecast")

    col1, col2 = st.columns(2)

    with col1:

        date_column = st.selectbox(
            "Select Date Column",
            date_columns
        )

    with col2:

        target_column = st.selectbox(
            "Select Target Column",
            numeric_columns
        )

    col3, col4 = st.columns(2)

    with col3:

        frequency = st.selectbox(
            "Forecast Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
            index=2
        )

    with col4:

        forecast_periods = st.number_input(
            "Number of Future Periods",
            min_value=1,
            max_value=100,
            value=6,
            step=1
        )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column, target_column]
    )

    if data.empty:
        st.warning("No valid records available.")
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

    historical = (
        data.set_index(date_column)[target_column]
        .sort_index()
        .resample(resample_frequency)
        .mean()
        .dropna()
    )

    if len(historical) < 3:
        st.warning(
            "At least 3 historical periods are required "
            "to build a forecast."
        )
        return

    historical_df = historical.reset_index()

    historical_df.columns = [
        "Date",
        "Actual"
    ]

    # -----------------------------------------------------
    # CREATE MODEL
    # -----------------------------------------------------

    X = np.arange(
        len(historical_df)
    ).reshape(-1, 1)

    y = historical_df["Actual"].values

    model = LinearRegression()

    model.fit(X, y)

    # -----------------------------------------------------
    # MODEL EVALUATION
    # -----------------------------------------------------

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    # -----------------------------------------------------
    # FUTURE FORECAST
    # -----------------------------------------------------

    future_X = np.arange(
        len(historical_df),
        len(historical_df) + int(forecast_periods)
    ).reshape(-1, 1)

    future_predictions = model.predict(
        future_X
    )

    future_dates = pd.date_range(
        start=historical.index[-1],
        periods=int(forecast_periods) + 1,
        freq=resample_frequency
    )[1:]

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
        f"{r2:.3f}"
    )

    col4.metric(
        "Historical Periods",
        f"{len(historical_df):,}"
    )

    st.caption(
        "These metrics evaluate the model on the same historical "
        "data used for training. They are not a reliable estimate "
        "of future forecasting accuracy."
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
            name="Historical Data"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=historical_df["Date"],
            y=predictions,
            mode="lines",
            name="Model Fit",
            line=dict(dash="dot")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted"],
            mode="lines+markers",
            name="Future Forecast",
            line=dict(dash="dash")
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
        use_container_width=True
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
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # FORECAST SUMMARY
    # -----------------------------------------------------

    st.markdown("### Forecast Summary")

    first_prediction = forecast_df["Predicted"].iloc[0]

    last_prediction = forecast_df["Predicted"].iloc[-1]

    forecast_change = (
        last_prediction - first_prediction
    )

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
    ).encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        data=export_csv,
        file_name="forecast_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "DataFlow AI | Forecasting & Prediction"
    )


import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# TREND ANALYSIS CONFIGURATION
# =========================================================

TEAL = "#0f9f93"
NAVY = "#14233f"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_dates(series):
    """Convert a Series into datetime values, supporting mixed formats."""

    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )

    except (TypeError, ValueError):
        # Compatibility fallback for older Pandas versions
        return pd.to_datetime(
            series,
            errors="coerce"
        )


def detect_date_columns(df):
    """Automatically detect columns containing valid date values."""

    date_columns = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue

        # Avoid interpreting ordinary numeric columns as dates
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        parsed = parse_dates(df[col])

        valid_ratio = parsed.notna().mean()
        unique_dates = parsed.nunique()

        if valid_ratio >= 0.80 and unique_dates >= 2:
            date_columns.append(col)

    return date_columns


def get_numeric_columns(df):
    """Detect numeric columns, including numbers stored as text."""

    numeric_columns = []

    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(col)

    return numeric_columns


def prepare_trend_data(df, date_column, value_column):
    """Prepare date and numeric columns for trend analysis."""

    trend_data = df[[date_column, value_column]].copy()

    trend_data[date_column] = parse_dates(
        trend_data[date_column]
    )

    trend_data[value_column] = pd.to_numeric(
        trend_data[value_column],
        errors="coerce"
    )

    return trend_data


def aggregate_trend_data(data, date_column, value_column,
                         frequency, aggregation):
    """Aggregate numerical values by the selected time frequency."""

    grouped = data.set_index(date_column).sort_index()

    resampler = grouped[value_column].resample(frequency)

    if aggregation == "Sum":
        result = resampler.sum(min_count=1)

    elif aggregation == "Average":
        result = resampler.mean()

    elif aggregation == "Count":
        result = resampler.count()

    elif aggregation == "Minimum":
        result = resampler.min()

    else:
        result = resampler.max()

    trend_df = result.reset_index()

    trend_df.columns = [
        "Date",
        "Value"
    ]

    return trend_df.dropna(subset=["Value"])


def calculate_trend_metrics(trend_df):
    """Calculate overall trend metrics."""

    first_value = float(trend_df["Value"].iloc[0])
    last_value = float(trend_df["Value"].iloc[-1])

    total_change = last_value - first_value

    if first_value != 0:

        percentage_change = (
            total_change / abs(first_value)
        ) * 100

    else:
        percentage_change = None

    average_value = trend_df["Value"].mean()

    if last_value > first_value:
        trend_direction = "Increasing"

    elif last_value < first_value:
        trend_direction = "Decreasing"

    else:
        trend_direction = "Stable"

    return (
        first_value,
        last_value,
        total_change,
        percentage_change,
        average_value,
        trend_direction
    )


# =========================================================
# MAIN TREND ANALYSIS PAGE
# =========================================================

def render_trend_analysis(df):

    st.markdown(
        '<p class="eyebrow">TIME SERIES & PATTERN ANALYSIS</p>',
        unsafe_allow_html=True
    )

    st.title("Trend Analysis")

    st.markdown(
        "Explore historical patterns, monitor changes over time, "
        "and identify growth or decline in your data."
    )

    # -----------------------------------------------------
    # VALIDATE DATA
    # -----------------------------------------------------

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    data = df.copy(deep=True)

    # -----------------------------------------------------
    # DETECT DATE AND NUMERIC COLUMNS
    # -----------------------------------------------------

    date_columns = detect_date_columns(data)

    numeric_columns = get_numeric_columns(data)

    if not date_columns:

        st.warning(
            "No suitable date column was detected. "
            "Please check your dataset's date format."
        )

        return

    if not numeric_columns:

        st.warning(
            "No numeric columns were detected for trend analysis."
        )

        return

    # -----------------------------------------------------
    # USER SELECTIONS
    # -----------------------------------------------------

    st.markdown("### Configure Your Analysis")

    col1, col2 = st.columns(2)

    with col1:

        date_column = st.selectbox(
            "Select Date Column",
            date_columns,
            key="trend_date_column"
        )

    with col2:

        value_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns,
            key="trend_numeric_column"
        )

    col3, col4, col5 = st.columns(3)

    with col3:

        frequency = st.selectbox(
            "Time Frequency",
            [
                "Daily",
                "Weekly",
                "Monthly",
                "Quarterly",
                "Yearly"
            ],
            index=2,
            key="trend_frequency"
        )

    with col4:

        aggregation = st.selectbox(
            "Aggregation",
            [
                "Sum",
                "Average",
                "Count",
                "Minimum",
                "Maximum"
            ],
            key="trend_aggregation"
        )

    with col5:

        chart_type = st.selectbox(
            "Chart Type",
            [
                "Line Chart",
                "Bar Chart",
                "Area Chart"
            ],
            key="trend_chart_type"
        )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    trend_data = prepare_trend_data(
        data,
        date_column,
        value_column
    )

    trend_data = trend_data.dropna(
        subset=[date_column]
    )

    if aggregation != "Count":

        trend_data = trend_data.dropna(
            subset=[value_column]
        )

    if trend_data.empty:

        st.warning(
            "No valid date and numeric records are available."
        )

        return

    # -----------------------------------------------------
    # FREQUENCY MAPPING
    # -----------------------------------------------------

    frequency_mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "MS",
        "Quarterly": "QS",
        "Yearly": "YS"
    }

    period = frequency_mapping[frequency]

    # -----------------------------------------------------
    # AGGREGATION
    # -----------------------------------------------------

    trend_df = aggregate_trend_data(
        trend_data,
        date_column,
        value_column,
        period,
        aggregation
    )

    if trend_df.empty:

        st.warning(
            "No results available for the selected frequency."
        )

        return

    # -----------------------------------------------------
    # TREND METRICS
    # -----------------------------------------------------

    (
        first_value,
        last_value,
        total_change,
        percentage_change,
        average_value,
        trend_direction
    ) = calculate_trend_metrics(trend_df)

    # -----------------------------------------------------
    # DISPLAY METRICS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Trend Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "First Period",
        f"{first_value:,.2f}"
    )

    col2.metric(
        "Latest Period",
        f"{last_value:,.2f}"
    )

    col3.metric(
        "Average Value",
        f"{average_value:,.2f}"
    )

    col4.metric(
        "Total Change",
        f"{total_change:+,.2f}"
    )

    if percentage_change is not None:

        st.caption(
            f"Percentage change from first to last period: "
            f"{percentage_change:+.2f}%"
        )

    else:

        st.caption(
            "Percentage change is unavailable because "
            "the first period has a value of zero."
        )

    st.info(
        f"Overall direction based on the first and last periods: "
        f"**{trend_direction}**"
    )

    # -----------------------------------------------------
    # TREND VISUALIZATION
    # -----------------------------------------------------

    st.markdown("### Trend Visualization")

    if chart_type == "Line Chart":

        fig = px.line(
            trend_df,
            x="Date",
            y="Value",
            markers=True,
            color_discrete_sequence=[TEAL],
            template="plotly_white"
        )

    elif chart_type == "Bar Chart":

        fig = px.bar(
            trend_df,
            x="Date",
            y="Value",
            color_discrete_sequence=[TEAL],
            template="plotly_white"
        )

    else:

        fig = px.area(
            trend_df,
            x="Date",
            y="Value",
            color_discrete_sequence=[TEAL],
            template="plotly_white"
        )

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title=value_column,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # -----------------------------------------------------
    # PERIOD-TO-PERIOD CHANGE
    # -----------------------------------------------------

    st.markdown("### Period-over-Period Change")

    change_df = trend_df.copy()

    change_df["Previous Value"] = (
        change_df["Value"].shift(1)
    )

    change_df["Change"] = (
        change_df["Value"] -
        change_df["Previous Value"]
    )

    change_df["Change Percentage"] = (
        change_df["Value"].pct_change(fill_method=None) * 100
    )

    # Avoid displaying infinite percentages when previous value is zero
    change_df["Change Percentage"] = (
        change_df["Change Percentage"].replace(
            [float("inf"), float("-inf")],
            float("nan")
        )
    )

    change_df = change_df.dropna(
        subset=["Previous Value"]
    ).copy()

    if not change_df.empty:

        fig = px.bar(
            change_df,
            x="Date",
            y="Change",
            color="Change",
            color_continuous_scale="RdYlGn",
            template="plotly_white"
        )

        fig.update_layout(
            height=350,
            xaxis_title="Date",
            yaxis_title="Change"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.dataframe(
            change_df.round(2),
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "At least two time periods are required "
            "for period-over-period comparison."
        )

    # -----------------------------------------------------
    # TREND SUMMARY
    # -----------------------------------------------------

    st.markdown("### Trend Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(f"- **Selected metric:** {value_column}")

        st.write(f"- **Time frequency:** {frequency}")

        st.write(f"- **Aggregation:** {aggregation}")

    with summary_col2:

        st.write(f"- **Periods analyzed:** {len(trend_df)}")

        st.write(f"- **Overall direction:** {trend_direction}")

        st.write(f"- **Date column:** {date_column}")

    # -----------------------------------------------------
    # DOWNLOAD RESULTS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Export Trend Results")

    export_df = trend_df.copy()

    export_csv = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Download Trend Analysis",
        data=export_csv,
        file_name="trend_analysis.csv",
        mime="text/csv",
        width="stretch"
    )

    st.caption(
        "DataFlow AI | Trend Analysis"
    )


import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# TREND ANALYSIS PAGE
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

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.80 and parsed.nunique() >= 2:
                date_columns.append(col)

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

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
            date_columns
        )

    with col2:

        value_column = st.selectbox(
            "Select Numeric Column",
            numeric_columns
        )

    col3, col4, col5 = st.columns(3)

    with col3:

        frequency = st.selectbox(
            "Time Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
            index=2
        )

    with col4:

        aggregation = st.selectbox(
            "Aggregation",
            ["Sum", "Average", "Count", "Minimum", "Maximum"]
        )

    with col5:

        chart_type = st.selectbox(
            "Chart Type",
            ["Line Chart", "Bar Chart", "Area Chart"]
        )

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column]
    )

    if aggregation != "Count":

        data = data.dropna(
            subset=[value_column]
        )

    if data.empty:
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

    grouped = data.set_index(
        date_column
    ).sort_index()

    if aggregation == "Sum":

        result = grouped[value_column].resample(
            period
        ).sum()

    elif aggregation == "Average":

        result = grouped[value_column].resample(
            period
        ).mean()

    elif aggregation == "Count":

        result = grouped[value_column].resample(
            period
        ).count()

    elif aggregation == "Minimum":

        result = grouped[value_column].resample(
            period
        ).min()

    else:

        result = grouped[value_column].resample(
            period
        ).max()

    trend_df = result.reset_index()

    trend_df.columns = [
        "Date",
        "Value"
    ]

    trend_df = trend_df.dropna(
        subset=["Value"]
    )

    if trend_df.empty:
        st.warning(
            "No results available for the selected frequency."
        )
        return

    # -----------------------------------------------------
    # TREND METRICS
    # -----------------------------------------------------

    first_value = float(
        trend_df["Value"].iloc[0]
    )

    last_value = float(
        trend_df["Value"].iloc[-1]
    )

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
            template="plotly_white"
        )

    elif chart_type == "Bar Chart":

        fig = px.bar(
            trend_df,
            x="Date",
            y="Value",
            template="plotly_white"
        )

    else:

        fig = px.area(
            trend_df,
            x="Date",
            y="Value",
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
        use_container_width=True
    )

    # -----------------------------------------------------
    # PERIOD-TO-PERIOD CHANGE
    # -----------------------------------------------------

    st.markdown("### Period-over-Period Change")

    trend_df["Previous Value"] = (
        trend_df["Value"].shift(1)
    )

    trend_df["Change"] = (
        trend_df["Value"] -
        trend_df["Previous Value"]
    )

    trend_df["Change Percentage"] = (
        trend_df["Value"].pct_change() * 100
    )

    change_df = trend_df.dropna(
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
            use_container_width=True
        )

        st.dataframe(
            change_df.round(2),
            use_container_width=True,
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

    st.write(
        f"- **Selected metric:** {value_column}"
    )

    st.write(
        f"- **Time frequency:** {frequency}"
    )

    st.write(
        f"- **Aggregation:** {aggregation}"
    )

    st.write(
        f"- **Periods analyzed:** {len(trend_df)}"
    )

    st.write(
        f"- **Overall direction:** {trend_direction}"
    )

    # -----------------------------------------------------
    # DOWNLOAD RESULTS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Export Trend Results")

    export_df = trend_df.copy()

    export_csv = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Trend Analysis",
        data=export_csv,
        file_name="trend_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "DataFlow AI | Trend Analysis"
    )

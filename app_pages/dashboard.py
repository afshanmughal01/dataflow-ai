
import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# DASHBOARD CONFIGURATION
# =========================================================

NAVY = "#14233f"
TEAL = "#0f9f93"
MUTED = "#64748b"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def detect_date_columns(df):
    """Detect columns containing valid date values."""

    date_columns = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue

        # Avoid interpreting ordinary numeric columns as dates
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        converted = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.80 and converted.notna().sum() >= 2:
            date_columns.append(col)

    return date_columns


def get_numeric_columns(df):
    """Return numeric columns with at least one valid value."""

    return [
        col for col in df.select_dtypes(include="number").columns
        if df[col].notna().any()
    ]


def get_categorical_columns(df):
    """Return suitable categorical columns."""

    categorical_columns = []

    for col in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        unique_count = df[col].nunique(dropna=True)

        if 2 <= unique_count <= 30:
            categorical_columns.append(col)

    return categorical_columns


def format_number(value):
    """Format large numbers for KPI cards."""

    return f"{value:,.0f}"


# =========================================================
# MAIN DASHBOARD
# =========================================================

def render_dashboard(df):

    st.markdown(
        '<p class="eyebrow">DATA INTELLIGENCE PLATFORM</p>',
        unsafe_allow_html=True
    )

    st.title("Executive Dashboard")

    st.markdown(
        "Explore your dataset, monitor data quality, "
        "and discover important patterns."
    )

    if df is None or df.empty:
        st.warning("No dataset available. Please upload a dataset first.")
        return

    # Work with a copy so dashboard filters never modify saved data
    data = df.copy()

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.markdown("### Dataset Overview")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.caption("Dataset records")
        st.write(f"**{len(data):,} rows**")

    with info_col2:
        st.caption("Available fields")
        st.write(f"**{len(data.columns):,} columns**")

    with info_col3:
        st.caption("Numeric fields")
        st.write(f"**{len(get_numeric_columns(data)):,} columns**")

    st.divider()

    # -----------------------------------------------------
    # DYNAMIC FILTERS
    # -----------------------------------------------------

    st.markdown("### Dataset Filters")

    date_columns = detect_date_columns(data)
    categorical_columns = get_categorical_columns(data)

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        if date_columns:

            selected_date = st.selectbox(
                "Select date column",
                date_columns,
                key="dashboard_date_column"
            )

            parsed_dates = pd.to_datetime(
                data[selected_date],
                errors="coerce"
            )

            valid_dates = parsed_dates.dropna()

            if not valid_dates.empty:

                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="dashboard_date_range"
                )

                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

                    start_date, end_date = date_range

                    data = data[
                        parsed_dates.dt.date.between(
                            start_date,
                            end_date
                        )
                    ]

        else:
            st.info("No suitable date column detected.")

    with filter_col2:

        if categorical_columns:

            selected_category = st.selectbox(
                "Filter by category",
                ["All"] + categorical_columns,
                key="dashboard_category_column"
            )

            if selected_category != "All":

                available_values = sorted(
                    data[selected_category]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_values = st.multiselect(
                    "Select values",
                    available_values,
                    default=available_values,
                    key="dashboard_category_values"
                )

                data = data[
                    data[selected_category]
                    .astype(str)
                    .isin(selected_values)
                ]

        else:
            st.info("No suitable category column detected.")

    # -----------------------------------------------------
    # HANDLE EMPTY FILTER RESULTS
    # -----------------------------------------------------

    if data.empty:

        st.warning(
            "No records match the selected filters. "
            "Please adjust your filters."
        )
        return

    st.divider()

    # -----------------------------------------------------
    # KEY PERFORMANCE INDICATORS
    # -----------------------------------------------------

    st.markdown("### Key Performance Indicators")

    total_rows = len(data)
    total_columns = len(data.columns)

    total_cells = data.size

    missing_cells = int(data.isna().sum().sum())

    completeness = (
        ((total_cells - missing_cells) / total_cells) * 100
        if total_cells > 0 else 0
    )

    duplicate_rows = int(data.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        format_number(total_rows)
    )

    col2.metric(
        "Total Columns",
        format_number(total_columns)
    )

    col3.metric(
        "Data Completeness",
        f"{completeness:.1f}%"
    )

    col4.metric(
        "Duplicate Records",
        format_number(duplicate_rows)
    )

    st.divider()

    # -----------------------------------------------------
    # AUTOMATED DATA INSIGHTS
    # -----------------------------------------------------

    st.markdown("### Automated Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.markdown("**Data Quality**")

        if completeness >= 95:
            st.success(
                f"Excellent completeness: {completeness:.1f}% "
                "of cells contain values."
            )

        elif completeness >= 80:
            st.info(
                f"Data completeness is {completeness:.1f}%. "
                "Some missing values need attention."
            )

        else:
            st.warning(
                f"Data completeness is {completeness:.1f}%. "
                "Consider cleaning missing values."
            )

    with insight_col2:

        st.markdown("**Duplicate Records**")

        if duplicate_rows == 0:
            st.success("No duplicate records detected.")

        else:
            st.warning(
                f"{duplicate_rows:,} duplicate records found. "
                "Review them in Data Preparation."
            )

    # -----------------------------------------------------
    # VISUAL ANALYTICS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Visual Analytics")

    numeric_columns = get_numeric_columns(data)
    categorical_columns = get_categorical_columns(data)

    chart_col1, chart_col2 = st.columns(2)

    # -----------------------------------------------------
    # CATEGORY DISTRIBUTION
    # -----------------------------------------------------

    with chart_col1:

        st.markdown("**Category Distribution**")

        if categorical_columns:

            category_column = st.selectbox(
                "Choose category",
                categorical_columns,
                key="dashboard_category_chart"
            )

            category_data = (
                data[category_column]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            category_data.columns = [
                "Category",
                "Count"
            ]

            fig = px.bar(
                category_data,
                x="Category",
                y="Count",
                color_discrete_sequence=[TEAL],
                template="plotly_white"
            )

            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "No suitable categorical columns available."
            )

    # -----------------------------------------------------
    # NUMERICAL DISTRIBUTION
    # -----------------------------------------------------

    with chart_col2:

        st.markdown("**Numerical Distribution**")

        if numeric_columns:

            numeric_column = st.selectbox(
                "Choose numerical column",
                numeric_columns,
                key="dashboard_numeric_chart"
            )

            fig = px.histogram(
                data,
                x=numeric_column,
                nbins=30,
                color_discrete_sequence=[TEAL],
                template="plotly_white"
            )

            fig.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "No numerical columns available for distribution analysis."
            )

    # -----------------------------------------------------
    # TIME SERIES TREND
    # -----------------------------------------------------

    if date_columns and numeric_columns:

        st.divider()

        st.markdown("### Time-Based Analysis")

        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:

            trend_date = st.selectbox(
                "Date column",
                date_columns,
                key="dashboard_trend_date"
            )

        with trend_col2:

            trend_numeric = st.selectbox(
                "Numerical column",
                numeric_columns,
                key="dashboard_trend_numeric"
            )

        trend_data = data[[trend_date, trend_numeric]].copy()

        trend_data[trend_date] = pd.to_datetime(
            trend_data[trend_date],
            errors="coerce"
        )

        trend_data[trend_numeric] = pd.to_numeric(
            trend_data[trend_numeric],
            errors="coerce"
        )

        trend_data = trend_data.dropna()

        if not trend_data.empty:

            trend_data = trend_data.sort_values(trend_date)

            trend_data = (
                trend_data
                .set_index(trend_date)[trend_numeric]
                .resample("ME")
                .sum()
                .reset_index()
            )

            fig = px.line(
                trend_data,
                x=trend_date,
                y=trend_numeric,
                markers=True,
                color_discrete_sequence=[TEAL],
                template="plotly_white"
            )

            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title=trend_numeric
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Not enough valid data to generate a trend chart.")

    # -----------------------------------------------------
    # DASHBOARD SUMMARY
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Dashboard Summary")

    st.write(
        f"""
        Your current filtered dataset contains **{total_rows:,} records**
        and **{total_columns:,} columns**.

        Data completeness is **{completeness:.1f}%**,
        with **{duplicate_rows:,} duplicate records** detected.
        """
    )

    st.caption(
        "DataFlow AI | Python Analytics Studio"
    )

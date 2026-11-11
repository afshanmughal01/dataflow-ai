
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# =========================================================
# BUSINESS INSIGHTS CONFIGURATION
# =========================================================

TEAL = "#0f9f93"
NAVY = "#14233f"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_numeric_columns(df):
    """Detect numeric columns, including numbers stored as text."""

    numeric_columns = []

    for column in df.columns:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(column)

    return numeric_columns


def get_categorical_columns(df):
    """Detect categorical columns."""

    categorical_columns = []

    for column in df.columns:

        if (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[column])
        ):
            categorical_columns.append(column)

    return categorical_columns


def prepare_numeric_series(series):
    """Convert a Series into valid numeric values."""

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def format_number(value):
    """Format numeric values for display."""

    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def prepare_category_data(df, category_column, metric_column):
    """Prepare category and metric data for comparison."""

    category_data = df[
        [category_column, metric_column]
    ].copy()

    category_data[metric_column] = prepare_numeric_series(
        category_data[metric_column]
    )

    category_data[category_column] = (
        category_data[category_column]
        .astype("string")
        .fillna("Missing")
    )

    category_data = category_data.dropna(
        subset=[metric_column]
    )

    return category_data


def calculate_correlation(df, numeric_columns):
    """Calculate correlations using valid, non-constant numeric columns."""

    numeric_data = pd.DataFrame(index=df.index)

    for column in numeric_columns:

        numeric_data[column] = prepare_numeric_series(
            df[column]
        )

    # Remove columns with no valid numeric values
    numeric_data = numeric_data.dropna(
        axis=1,
        how="all"
    )

    # Remove constant columns
    valid_columns = [
        column
        for column in numeric_data.columns
        if numeric_data[column].nunique(dropna=True) > 1
    ]

    numeric_data = numeric_data[valid_columns]

    correlation = numeric_data.corr(
        method="pearson"
    )

    return numeric_data, correlation


def get_correlation_pairs(correlation):
    """Get unique metric relationships."""

    pairs = []

    columns = correlation.columns.tolist()

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            column_a = columns[i]
            column_b = columns[j]

            value = correlation.loc[
                column_a,
                column_b
            ]

            if pd.notna(value):

                pairs.append({
                    "Metric 1": column_a,
                    "Metric 2": column_b,
                    "Correlation": value,
                    "Absolute Correlation": abs(value)
                })

    if not pairs:
        return pd.DataFrame(
            columns=[
                "Metric 1",
                "Metric 2",
                "Correlation",
                "Absolute Correlation"
            ]
        )

    return (
        pd.DataFrame(pairs)
        .sort_values(
            "Absolute Correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )


# =========================================================
# MAIN BUSINESS INSIGHTS PAGE
# =========================================================

def render_business_insights(df):

    st.markdown(
        '<p class="eyebrow">BUSINESS INTELLIGENCE</p>',
        unsafe_allow_html=True
    )

    st.title("Business Insights")

    st.markdown(
        "Discover meaningful patterns, identify top-performing "
        "categories, and turn your data into actionable insights."
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

    numeric_columns = get_numeric_columns(data)

    categorical_columns = get_categorical_columns(data)

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(data):,}"
    )

    col2.metric(
        "Total Columns",
        f"{len(data.columns):,}"
    )

    col3.metric(
        "Numeric Fields",
        f"{len(numeric_columns):,}"
    )

    col4.metric(
        "Categorical Fields",
        f"{len(categorical_columns):,}"
    )

    st.divider()

    # -----------------------------------------------------
    # NUMERIC BUSINESS ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 1. Numerical Performance Analysis")

    if numeric_columns:

        selected_metric = st.selectbox(
            "Select a business metric",
            numeric_columns,
            key="business_numeric_metric"
        )

        metric_data = prepare_numeric_series(
            data[selected_metric]
        ).dropna()

        if not metric_data.empty:

            total_value = metric_data.sum()
            average_value = metric_data.mean()
            maximum_value = metric_data.max()
            minimum_value = metric_data.min()
            median_value = metric_data.median()
            standard_deviation = metric_data.std()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total",
                format_number(total_value)
            )

            col2.metric(
                "Average",
                format_number(average_value)
            )

            col3.metric(
                "Maximum",
                format_number(maximum_value)
            )

            col4.metric(
                "Minimum",
                format_number(minimum_value)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Metric Distribution**")

                distribution_data = pd.DataFrame({
                    selected_metric: metric_data
                })

                fig = px.histogram(
                    distribution_data,
                    x=selected_metric,
                    nbins=30,
                    template="plotly_white",
                    color_discrete_sequence=[TEAL]
                )

                fig.update_layout(
                    height=350,
                    xaxis_title=selected_metric,
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            with col2:

                st.markdown("**Metric Summary**")

                summary = pd.DataFrame({
                    "Measure": [
                        "Total",
                        "Average",
                        "Median",
                        "Maximum",
                        "Minimum",
                        "Standard Deviation"
                    ],
                    "Value": [
                        total_value,
                        average_value,
                        median_value,
                        maximum_value,
                        minimum_value,
                        standard_deviation
                    ]
                })

                st.dataframe(
                    summary.round(2),
                    width="stretch",
                    hide_index=True
                )

        else:

            st.info(
                "The selected metric does not contain valid numeric values."
            )

    else:

        st.info(
            "No numeric columns are available for numerical analysis."
        )

    st.divider()

    # -----------------------------------------------------
    # CATEGORY PERFORMANCE
    # -----------------------------------------------------

    st.markdown("### 2. Category Performance Analysis")

    if categorical_columns and numeric_columns:

        col1, col2 = st.columns(2)

        with col1:

            category_column = st.selectbox(
                "Select Category",
                categorical_columns,
                key="business_category_column"
            )

        with col2:

            category_metric = st.selectbox(
                "Select Performance Metric",
                numeric_columns,
                key="business_category_metric"
            )

        category_data = prepare_category_data(
            data,
            category_column,
            category_metric
        )

        if not category_data.empty:

            performance = (
                category_data
                .groupby(
                    category_column,
                    dropna=False,
                    observed=True
                )[category_metric]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            total_categories = len(performance)

            performance = performance.head(15)

            fig = px.bar(
                performance,
                x=category_column,
                y=category_metric,
                color=category_metric,
                color_continuous_scale="Teal",
                template="plotly_white",
                title=f"{category_metric} by {category_column}"
            )

            fig.update_layout(
                height=400,
                showlegend=False,
                xaxis_title=category_column,
                yaxis_title=category_metric
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # Top category
            top_category = performance.iloc[0]

            st.success(
                f"Top category: **{top_category[category_column]}** "
                f"with {category_metric} of "
                f"**{top_category[category_metric]:,.2f}**."
            )

            st.caption(
                f"Showing the top {len(performance)} of "
                f"{total_categories:,} categories."
            )

            st.dataframe(
                performance.round(2),
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "No valid category and metric combinations were found."
            )

    else:

        st.info(
            "Category performance requires at least one categorical "
            "and one numeric column."
        )

    st.divider()

    # -----------------------------------------------------
    # CATEGORY DISTRIBUTION
    # -----------------------------------------------------

    st.markdown("### 3. Customer / Category Distribution")

    if categorical_columns:

        distribution_column = st.selectbox(
            "Select a column to analyze",
            categorical_columns,
            key="business_distribution_column"
        )

        distribution_values = (
            data[distribution_column]
            .astype("string")
            .fillna("Missing")
        )

        distribution = (
            distribution_values
            .value_counts()
            .head(15)
            .rename_axis("Category")
            .reset_index(name="Records")
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                distribution,
                names="Category",
                values="Records",
                hole=0.45,
                title="Category Share",
                template="plotly_white",
                color_discrete_sequence=px.colors.sequential.Teal
            )

            fig.update_layout(
                height=380
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.bar(
                distribution,
                x="Records",
                y="Category",
                orientation="h",
                title="Category Record Count",
                template="plotly_white",
                color_discrete_sequence=[TEAL]
            )

            fig.update_layout(
                height=380,
                yaxis={
                    "categoryorder": "total ascending"
                }
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        st.dataframe(
            distribution,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No categorical columns are available for distribution analysis."
        )

    st.divider()

    # -----------------------------------------------------
    # CORRELATION INSIGHTS
    # -----------------------------------------------------

    st.markdown("### 4. Relationship Between Business Metrics")

    if len(numeric_columns) >= 2:

        numeric_data, correlation = calculate_correlation(
            data,
            numeric_columns
        )

        if correlation.shape[1] >= 2:

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Metric Correlation Matrix",
                template="plotly_white"
            )

            fig.update_layout(
                height=500
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # Find strongest relationships
            pairs_df = get_correlation_pairs(
                correlation
            )

            if not pairs_df.empty:

                st.markdown(
                    "**Strongest Metric Relationships**"
                )

                st.dataframe(
                    pairs_df.drop(
                        columns=["Absolute Correlation"]
                    ).round(3).head(10),
                    width="stretch",
                    hide_index=True
                )

                strongest = pairs_df.iloc[0]

                relationship_type = (
                    "positive"
                    if strongest["Correlation"] > 0
                    else "negative"
                )

                st.info(
                    f"Strongest observed relationship: "
                    f"**{strongest['Metric 1']}** and "
                    f"**{strongest['Metric 2']}** "
                    f"have a {relationship_type} correlation of "
                    f"**{strongest['Correlation']:.2f}**. "
                    "Correlation does not establish causation."
                )

            else:

                st.info(
                    "No meaningful relationships could be calculated "
                    "between the selected numeric columns."
                )

        else:

            st.info(
                "At least two non-constant numeric columns with valid "
                "values are required for correlation analysis."
            )

    else:

        st.info(
            "At least two numeric columns are required for correlation analysis."
        )

    st.divider()

    # -----------------------------------------------------
    # AUTOMATIC BUSINESS OBSERVATIONS
    # -----------------------------------------------------

    st.markdown("### 5. Automatic Business Observations")

    observations = []

    total_cells = max(data.size, 1)

    missing_cells = int(
        data.isna().sum().sum()
    )

    missing_percentage = (
        missing_cells / total_cells
    ) * 100

    duplicate_count = int(
        data.duplicated().sum()
    )

    # Missing values
    if missing_percentage > 0:

        observations.append(
            f"Your dataset contains {missing_percentage:.1f}% "
            "missing cells. Review data completeness before making decisions."
        )

    else:

        observations.append(
            "No missing cells were detected in the dataset."
        )

    # Duplicate rows
    if duplicate_count > 0:

        observations.append(
            f"{duplicate_count:,} duplicate rows were found. "
            "Review them to avoid potentially misleading results."
        )

    else:

        observations.append(
            "No duplicate rows were detected."
        )

    # Numeric observations
    if numeric_columns:

        for column in numeric_columns:

            values = prepare_numeric_series(
                data[column]
            ).dropna()

            if values.empty:
                continue

            if (values < 0).any():

                observations.append(
                    f"Column '{column}' contains negative values. "
                    "Check whether negative values are valid for this metric."
                )

            if values.nunique() == 1:

                observations.append(
                    f"Column '{column}' has only one distinct numeric value."
                )

    # Display observations
    for index, observation in enumerate(
        observations,
        start=1
    ):

        st.write(
            f"**{index}.** {observation}"
        )

    st.caption(
        "These are automatically generated observations, "
        "not verified business conclusions."
    )

    # -----------------------------------------------------
    # EXPORT BUSINESS INSIGHTS
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Export Business Analysis")

    report_lines = [
        "DATAFLOW AI - BUSINESS INSIGHTS REPORT",
        "=" * 45,
        "",
        "DATASET OVERVIEW",
        f"Total Records: {len(data):,}",
        f"Total Columns: {len(data.columns):,}",
        f"Numeric Fields: {len(numeric_columns):,}",
        f"Categorical Fields: {len(categorical_columns):,}",
        f"Missing Cells: {missing_cells:,}",
        f"Missing Percentage: {missing_percentage:.2f}%",
        f"Duplicate Rows: {duplicate_count:,}",
        "",
        "AUTOMATIC OBSERVATIONS",
        "-" * 30
    ]

    for index, observation in enumerate(
        observations,
        start=1
    ):

        report_lines.append(
            f"{index}. {observation}"
        )

    report_text = "\n".join(report_lines)

    st.download_button(
        "Download Business Insights Report",
        data=report_text.encode("utf-8-sig"),
        file_name="business_insights_report.txt",
        mime="text/plain",
        width="stretch"
    )

    st.caption(
        "DataFlow AI | Business Insights"
    )

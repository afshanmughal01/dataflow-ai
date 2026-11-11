
import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# DATA ANALYSIS CONFIGURATION
# =========================================================

TEAL = "#0f9f93"
NAVY = "#14233f"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_numeric_columns(df):
    """Return numerical columns containing valid numeric values."""

    numeric_columns = []

    for col in df.columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.notna().any():
            numeric_columns.append(col)

    return numeric_columns


def get_categorical_columns(df):
    """Return categorical columns with suitable unique values."""

    categorical_columns = []

    for col in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        unique_count = df[col].nunique(dropna=True)

        if 2 <= unique_count <= 100:
            categorical_columns.append(col)

    return categorical_columns


def format_number(value):
    """Format numbers for display."""

    return f"{value:,.0f}"


def prepare_numeric_data(df, columns):
    """Convert selected columns to numeric values safely."""

    numeric_data = df[columns].copy()

    for col in columns:
        numeric_data[col] = pd.to_numeric(
            numeric_data[col],
            errors="coerce"
        )

    return numeric_data


# =========================================================
# MAIN DATA ANALYSIS PAGE
# =========================================================

def render_data_analysis(df):

    st.markdown(
        '<p class="eyebrow">ADVANCED ANALYTICS</p>',
        unsafe_allow_html=True
    )

    st.title("Data Analysis")

    st.markdown(
        "Explore statistical summaries, relationships, "
        "distributions, and patterns in your dataset."
    )

    # -----------------------------------------------------
    # VALIDATE DATASET
    # -----------------------------------------------------

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    data = df.copy(deep=True)

    numeric_columns = get_numeric_columns(data)

    categorical_columns = get_categorical_columns(data)

    numeric_data = prepare_numeric_data(
        data,
        numeric_columns
    )

    # -----------------------------------------------------
    # ANALYSIS OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Analysis Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        format_number(len(data))
    )

    col2.metric(
        "Numeric Columns",
        len(numeric_columns)
    )

    col3.metric(
        "Categorical Columns",
        len(categorical_columns)
    )

    st.divider()

    # -----------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------------------------------

    st.markdown("### 1. Descriptive Statistics")

    if numeric_columns:

        statistics = numeric_data.describe().T

        statistics["missing"] = numeric_data.isna().sum()

        statistics["median"] = numeric_data.median()

        statistics = statistics.rename(
            columns={
                "count": "Count",
                "mean": "Mean",
                "std": "Standard Deviation",
                "min": "Minimum",
                "25%": "25th Percentile",
                "50%": "50th Percentile",
                "75%": "75th Percentile",
                "max": "Maximum",
                "missing": "Missing Values",
                "median": "Median"
            }
        )

        st.dataframe(
            statistics.round(2),
            width="stretch"
        )

        csv_statistics = statistics.to_csv().encode("utf-8")

        st.download_button(
            "Download Statistical Summary",
            data=csv_statistics,
            file_name="statistical_summary.csv",
            mime="text/csv",
            width="stretch"
        )

    else:

        st.info(
            "No numerical columns are available for descriptive statistics."
        )

    st.divider()

    # -----------------------------------------------------
    # NUMERICAL DISTRIBUTION
    # -----------------------------------------------------

    st.markdown("### 2. Numerical Distribution")

    if numeric_columns:

        selected_numeric = st.selectbox(
            "Select numerical column",
            numeric_columns,
            key="analysis_distribution_column"
        )

        distribution_data = numeric_data[
            selected_numeric
        ].dropna()

        if not distribution_data.empty:

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.markdown("**Histogram**")

                fig = px.histogram(
                    x=distribution_data,
                    nbins=30,
                    labels={
                        "x": selected_numeric,
                        "count": "Frequency"
                    },
                    color_discrete_sequence=[TEAL],
                    template="plotly_white"
                )

                fig.update_layout(
                    height=350,
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            with chart_col2:

                st.markdown("**Box Plot**")

                fig = px.box(
                    y=distribution_data,
                    labels={
                        "y": selected_numeric
                    },
                    color_discrete_sequence=[TEAL],
                    template="plotly_white"
                )

                fig.update_layout(height=350)

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

        else:

            st.info(
                "This column does not contain valid numerical values."
            )

    else:

        st.info("Numerical distribution analysis is unavailable.")

    st.divider()

    # -----------------------------------------------------
    # CORRELATION ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 3. Correlation Analysis")

    if len(numeric_columns) >= 2:

        # Exclude columns with insufficient data or no variation
        correlation_data = numeric_data[
            numeric_columns
        ].dropna(axis=1, how="all")

        usable_columns = [
            col for col in correlation_data.columns
            if correlation_data[col].nunique() > 1
        ]

        correlation_data = correlation_data[
            usable_columns
        ]

        if len(usable_columns) >= 2:

            correlation = correlation_data.corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
                template="plotly_white"
            )

            fig.update_layout(
                height=max(400, len(usable_columns) * 55),
                coloraxis_colorbar_title="Correlation"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.caption(
                "Correlation ranges from -1 to +1. "
                "A positive value indicates a positive relationship, "
                "while a negative value indicates an inverse relationship. "
                "Correlation does not establish causation."
            )

            # Find strongest unique relationships
            pairs = []

            for i in range(len(usable_columns)):

                for j in range(i + 1, len(usable_columns)):

                    col_a = usable_columns[i]
                    col_b = usable_columns[j]

                    value = correlation.loc[col_a, col_b]

                    if pd.notna(value):

                        pairs.append({
                            "Column 1": col_a,
                            "Column 2": col_b,
                            "Correlation": value,
                            "Absolute Correlation": abs(value)
                        })

            if pairs:

                pair_df = pd.DataFrame(pairs)

                pair_df = pair_df.sort_values(
                    "Absolute Correlation",
                    ascending=False
                )

                st.markdown("#### Strongest Relationships")

                st.dataframe(
                    pair_df.drop(
                        columns=["Absolute Correlation"]
                    ).head(10).round(3),
                    width="stretch",
                    hide_index=True
                )

                strongest = pair_df.iloc[0]

                st.info(
                    f"Strongest relationship: "
                    f"**{strongest['Column 1']}** and "
                    f"**{strongest['Column 2']}** "
                    f"(Correlation: {strongest['Correlation']:.3f})"
                )

        else:

            st.info(
                "At least two numerical columns with varying values "
                "are required for correlation analysis."
            )

    else:

        st.info(
            "At least two numerical columns are required for correlation analysis."
        )

    st.divider()

    # -----------------------------------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 4. Categorical Analysis")

    if categorical_columns:

        selected_category = st.selectbox(
            "Select categorical column",
            categorical_columns,
            key="analysis_category_column"
        )

        category_counts = (
            data[selected_category]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(15)
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Count"
        ]

        category_col1, category_col2 = st.columns(2)

        with category_col1:

            st.markdown("**Category Frequency**")

            fig = px.bar(
                category_counts,
                x="Category",
                y="Count",
                color_discrete_sequence=[TEAL],
                template="plotly_white"
            )

            fig.update_layout(
                height=380,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with category_col2:

            st.markdown("**Category Proportion**")

            fig = px.pie(
                category_counts,
                names="Category",
                values="Count",
                hole=0.45,
                template="plotly_white"
            )

            fig.update_layout(height=380)

            st.plotly_chart(
                fig,
                width="stretch"
            )

        st.markdown("#### Category Frequency Table")

        st.dataframe(
            category_counts,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No categorical columns are available for this analysis."
        )

    st.divider()

    # -----------------------------------------------------
    # MISSING VALUE ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 5. Missing Value Analysis")

    missing_report = pd.DataFrame({
        "Column": data.columns,
        "Missing Values": data.isna().sum().values,
        "Missing Percentage": (
            data.isna().mean().values * 100
        )
    })

    missing_report = missing_report.sort_values(
        "Missing Percentage",
        ascending=False
    )

    missing_report = missing_report[
        missing_report["Missing Values"] > 0
    ]

    if not missing_report.empty:

        fig = px.bar(
            missing_report,
            x="Column",
            y="Missing Percentage",
            color_discrete_sequence=[TEAL],
            template="plotly_white"
        )

        fig.update_layout(
            height=350,
            yaxis_title="Missing Values (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.dataframe(
            missing_report.round(2),
            width="stretch",
            hide_index=True
        )

        total_missing = int(data.isna().sum().sum())

        st.warning(
            f"Your dataset contains {total_missing:,} missing cells "
            f"across {len(missing_report)} columns."
        )

    else:

        st.success("No missing values found in this dataset.")

    st.divider()

    # -----------------------------------------------------
    # ANALYSIS SUMMARY
    # -----------------------------------------------------

    st.markdown("### Analysis Summary")

    st.write(
        f"""
        Your dataset contains **{len(data):,} records** and
        **{len(data.columns):,} columns**.

        - Numerical columns: **{len(numeric_columns)}**
        - Categorical columns: **{len(categorical_columns)}**
        - Missing cells: **{int(data.isna().sum().sum()):,}**
        """
    )

    st.caption(
        "DataFlow AI | Statistical & Exploratory Analysis"
    )

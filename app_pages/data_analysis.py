
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


# =========================================================
# DATA ANALYSIS PAGE
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

    data = df.copy()

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # -----------------------------------------------------
    # ANALYSIS OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Analysis Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Numeric Columns", len(numeric_columns))
    col3.metric("Categorical Columns", len(categorical_columns))

    st.divider()

    # -----------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------------------------------

    st.markdown("### 1. Descriptive Statistics")

    if numeric_columns:

        statistics = data[numeric_columns].describe().T

        statistics["missing"] = data[numeric_columns].isna().sum()

        statistics["median"] = data[numeric_columns].median()

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
            use_container_width=True
        )

        csv_statistics = statistics.to_csv().encode("utf-8")

        st.download_button(
            "Download Statistical Summary",
            data=csv_statistics,
            file_name="statistical_summary.csv",
            mime="text/csv"
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

        distribution_data = pd.to_numeric(
            data[selected_numeric],
            errors="coerce"
        ).dropna()

        if not distribution_data.empty:

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.markdown("**Histogram**")

                fig = px.histogram(
                    distribution_data,
                    x=selected_numeric,
                    nbins=30,
                    color_discrete_sequence=["#0f9f93"],
                    template="plotly_white"
                )

                fig.update_layout(
                    height=350,
                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with chart_col2:

                st.markdown("**Box Plot**")

                fig = px.box(
                    distribution_data,
                    y=selected_numeric,
                    color_discrete_sequence=["#0f9f93"],
                    template="plotly_white"
                )

                fig.update_layout(height=350)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            st.info("This column does not contain valid numerical values.")

    else:

        st.info("Numerical distribution analysis is unavailable.")

    st.divider()

    # -----------------------------------------------------
    # CORRELATION ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 3. Correlation Analysis")

    if len(numeric_columns) >= 2:

        correlation = data[numeric_columns].corr()

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
            height=max(400, len(numeric_columns) * 55),
            coloraxis_colorbar_title="Correlation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Correlation ranges from -1 to +1. "
            "A positive value indicates a positive relationship, "
            "while a negative value indicates an inverse relationship. "
            "Correlation does not establish causation."
        )

        # Strongest unique relationship
        upper_triangle = correlation.where(
            pd.Series(
                True,
                index=correlation.index
            ).to_frame().dot(
                pd.Series(
                    True,
                    index=correlation.columns
                ).to_frame().T
            ).astype(bool)
        )

        pairs = []

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):

                col_a = numeric_columns[i]
                col_b = numeric_columns[j]

                value = correlation.loc[col_a, col_b]

                if pd.notna(value):
                    pairs.append({
                        "Column 1": col_a,
                        "Column 2": col_b,
                        "Correlation": value
                    })

        if pairs:

            pair_df = pd.DataFrame(pairs)

            pair_df["Absolute Correlation"] = (
                pair_df["Correlation"].abs()
            )

            pair_df = pair_df.sort_values(
                "Absolute Correlation",
                ascending=False
            )

            st.markdown("#### Strongest Relationships")

            st.dataframe(
                pair_df.drop(
                    columns=["Absolute Correlation"]
                ).head(10).round(3),
                use_container_width=True,
                hide_index=True
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
                color_discrete_sequence=["#0f9f93"],
                template="plotly_white"
            )

            fig.update_layout(
                height=380,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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
                use_container_width=True
            )

        st.markdown("#### Category Frequency Table")

        st.dataframe(
            category_counts,
            use_container_width=True,
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
            color_discrete_sequence=["#0f9f93"],
            template="plotly_white"
        )

        fig.update_layout(
            height=350,
            yaxis_title="Missing Values (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            missing_report.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success("No missing values found in this dataset.")

    st.divider()

    st.caption("DataFlow AI | Statistical & Exploratory Analysis")

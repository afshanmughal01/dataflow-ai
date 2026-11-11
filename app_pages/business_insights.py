
import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# BUSINESS INSIGHTS PAGE
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
    # DATASET OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(data):,}")

    col2.metric("Total Columns", f"{len(data.columns):,}")

    col3.metric("Numeric Fields", f"{len(numeric_columns):,}")

    col4.metric("Categorical Fields", f"{len(categorical_columns):,}")

    st.divider()

    # -----------------------------------------------------
    # NUMERIC BUSINESS ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 1. Numerical Performance Analysis")

    if numeric_columns:

        selected_metric = st.selectbox(
            "Select a business metric",
            numeric_columns
        )

        metric_data = pd.to_numeric(
            data[selected_metric],
            errors="coerce"
        ).dropna()

        if not metric_data.empty:

            total_value = metric_data.sum()
            average_value = metric_data.mean()
            maximum_value = metric_data.max()
            minimum_value = metric_data.min()
            median_value = metric_data.median()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total",
                f"{total_value:,.2f}"
            )

            col2.metric(
                "Average",
                f"{average_value:,.2f}"
            )

            col3.metric(
                "Maximum",
                f"{maximum_value:,.2f}"
            )

            col4.metric(
                "Minimum",
                f"{minimum_value:,.2f}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Metric Distribution**")

                fig = px.histogram(
                    metric_data,
                    x=selected_metric,
                    nbins=30,
                    template="plotly_white"
                )

                fig.update_layout(height=350)

                st.plotly_chart(
                    fig,
                    use_container_width=True
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
                        metric_data.std()
                    ]
                })

                st.dataframe(
                    summary.round(2),
                    use_container_width=True,
                    hide_index=True
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
                categorical_columns
            )

        with col2:

            category_metric = st.selectbox(
                "Select Performance Metric",
                numeric_columns,
                key="business_category_metric"
            )

        category_data = data[
            [category_column, category_metric]
        ].copy()

        category_data[category_metric] = pd.to_numeric(
            category_data[category_metric],
            errors="coerce"
        )

        category_data = category_data.dropna(
            subset=[category_column, category_metric]
        )

        if not category_data.empty:

            performance = (
                category_data
                .groupby(category_column)[category_metric]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

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
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if len(performance) > 0:

                top_category = performance.iloc[0]

                st.success(
                    f"Top category: **{top_category[category_column]}** "
                    f"with {category_metric} of "
                    f"**{top_category[category_metric]:,.2f}**."
                )

            st.dataframe(
                performance,
                use_container_width=True,
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

        distribution = (
            data[distribution_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(15)
            .reset_index()
        )

        distribution.columns = [
            "Category",
            "Records"
        ]

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                distribution,
                names="Category",
                values="Records",
                hole=0.45,
                title="Category Share",
                template="plotly_white"
            )

            fig.update_layout(height=380)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.bar(
                distribution,
                x="Records",
                y="Category",
                orientation="h",
                title="Category Record Count",
                template="plotly_white"
            )

            fig.update_layout(
                height=380,
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
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

        correlation = data[numeric_columns].corr(
            numeric_only=True
        )

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Metric Correlation Matrix",
            template="plotly_white"
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Find strongest unique relationship
        pairs = []

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                col_a = numeric_columns[i]
                col_b = numeric_columns[j]

                value = correlation.loc[col_a, col_b]

                if pd.notna(value):

                    pairs.append({
                        "Metric 1": col_a,
                        "Metric 2": col_b,
                        "Correlation": value
                    })

        if pairs:

            pairs_df = pd.DataFrame(pairs)

            pairs_df["Absolute Correlation"] = (
                pairs_df["Correlation"].abs()
            )

            pairs_df = pairs_df.sort_values(
                "Absolute Correlation",
                ascending=False
            )

            st.markdown("**Strongest Metric Relationships**")

            st.dataframe(
                pairs_df.drop(
                    columns=["Absolute Correlation"]
                ).round(3).head(10),
                use_container_width=True,
                hide_index=True
            )

            strongest = pairs_df.iloc[0]

            st.info(
                f"Strongest observed relationship: "
                f"**{strongest['Metric 1']}** and "
                f"**{strongest['Metric 2']}** "
                f"(correlation: {strongest['Correlation']:.2f}). "
                "Correlation does not establish causation."
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

    missing_percentage = (
        data.isna().sum().sum() /
        max(data.size, 1)
    ) * 100

    duplicate_count = int(data.duplicated().sum())

    if missing_percentage > 0:

        observations.append(
            f"Your dataset contains {missing_percentage:.1f}% "
            "missing cells. Review data completeness before making decisions."
        )

    else:

        observations.append(
            "No missing cells were detected in the dataset."
        )

    if duplicate_count > 0:

        observations.append(
            f"{duplicate_count:,} duplicate rows were found. "
            "Review them to avoid potentially misleading results."
        )

    else:

        observations.append(
            "No duplicate rows were detected."
        )

    if numeric_columns:

        for column in numeric_columns:

            values = pd.to_numeric(
                data[column],
                errors="coerce"
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
        "",
        f"Total Records: {len(data)}",
        f"Total Columns: {len(data.columns)}",
        f"Missing Cells: {int(data.isna().sum().sum())}",
        f"Duplicate Rows: {duplicate_count}",
        "",
        "AUTOMATIC OBSERVATIONS"
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
        data=report_text,
        file_name="business_insights_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.caption(
        "DataFlow AI | Business Insights"
    )

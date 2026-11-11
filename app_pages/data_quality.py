
import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# DATA QUALITY PAGE
# =========================================================

def render_data_quality(df):

    st.markdown(
        '<p class="eyebrow">DATA VALIDATION & MONITORING</p>',
        unsafe_allow_html=True
    )

    st.title("Data Quality")

    st.markdown(
        "Evaluate dataset reliability, identify inconsistencies, "
        "and discover areas that require cleaning."
    )

    # -----------------------------------------------------
    # VALIDATE DATASET
    # -----------------------------------------------------

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    data = df.copy()

    total_rows = len(data)
    total_columns = len(data.columns)
    total_cells = data.size

    # -----------------------------------------------------
    # QUALITY METRICS
    # -----------------------------------------------------

    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())

    completeness = (
        (1 - missing_cells / total_cells) * 100
        if total_cells else 0
    )

    uniqueness = (
        (1 - duplicate_rows / total_rows) * 100
        if total_rows else 0
    )

    # Basic completeness and uniqueness score
    quality_score = (completeness + uniqueness) / 2

    st.markdown("### Quality Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Quality Score",
        f"{quality_score:.1f}%"
    )

    col2.metric(
        "Completeness",
        f"{completeness:.1f}%"
    )

    col3.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )

    # Quality score visualization
    st.progress(
        int(round(quality_score)),
        text=f"Overall quality indicator: {quality_score:.1f}%"
    )

    st.caption(
        "This is a basic indicator based on completeness and row uniqueness, "
        "not a universal data-quality standard."
    )

    st.divider()

    # -----------------------------------------------------
    # MISSING VALUE ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 1. Missing Value Analysis")

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

    if missing_cells > 0:

        affected_columns = missing_report[
            missing_report["Missing Values"] > 0
        ]

        fig = px.bar(
            affected_columns,
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
            affected_columns.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success("No missing values detected.")

    st.divider()

    # -----------------------------------------------------
    # DUPLICATE ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 2. Duplicate Records")

    duplicate_data = data[
        data.duplicated(keep=False)
    ]

    if not duplicate_data.empty:

        st.warning(
            f"{duplicate_rows:,} duplicate rows were detected."
        )

        st.dataframe(
            duplicate_data.head(100),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Review duplicate records before removing them. "
            "Some repeated records may be valid."
        )

    else:

        st.success("No duplicate rows detected.")

    st.divider()

    # -----------------------------------------------------
    # COLUMN DATA TYPE ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 3. Column Data Types")

    dtype_report = pd.DataFrame({
        "Column": data.columns.astype(str),
        "Data Type": [
            str(dtype) for dtype in data.dtypes
        ],
        "Non-Missing Values": data.notna().sum().values,
        "Missing Values": data.isna().sum().values,
        "Unique Values": data.nunique(dropna=True).values
    })

    st.dataframe(
        dtype_report,
        use_container_width=True,
        hide_index=True
    )

    # Data type distribution
    dtype_counts = (
        data.dtypes.astype(str)
        .value_counts()
        .reset_index()
    )

    dtype_counts.columns = [
        "Data Type",
        "Column Count"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Data Type Distribution**")

        fig = px.pie(
            dtype_counts,
            names="Data Type",
            values="Column Count",
            hole=0.45,
            template="plotly_white"
        )

        fig.update_layout(height=330)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("**Column Information**")

        st.write(
            f"Total columns: **{total_columns}**"
        )

        for dtype, count in dtype_counts.values:

            st.write(
                f"• {dtype}: **{count} columns**"
            )

    st.divider()

    # -----------------------------------------------------
    # CONSTANT COLUMN DETECTION
    # -----------------------------------------------------

    st.markdown("### 4. Constant Columns")

    constant_columns = [
        col for col in data.columns
        if data[col].nunique(dropna=False) <= 1
    ]

    if constant_columns:

        st.warning(
            f"{len(constant_columns)} columns contain "
            "only one distinct value."
        )

        st.write(constant_columns)

        st.caption(
            "These columns may provide little analytical value."
        )

    else:

        st.success("No constant columns detected.")

    st.divider()

    # -----------------------------------------------------
    # HIGH CARDINALITY DETECTION
    # -----------------------------------------------------

    st.markdown("### 5. High Cardinality Columns")

    cardinality_report = pd.DataFrame({
        "Column": data.columns,
        "Unique Values": data.nunique(dropna=True).values,
        "Unique Percentage": (
            data.nunique(dropna=True).values /
            max(total_rows, 1) * 100
        )
    })

    high_cardinality = cardinality_report[
        cardinality_report["Unique Percentage"] >= 90
    ]

    if not high_cardinality.empty:

        st.info(
            f"{len(high_cardinality)} columns have "
            "90% or more unique values."
        )

        st.dataframe(
            high_cardinality.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "High cardinality can be expected for identifiers, "
            "emails, transaction IDs, and similar fields."
        )

    else:

        st.success(
            "No columns meet the high-cardinality threshold."
        )

    st.divider()

    # -----------------------------------------------------
    # QUALITY RECOMMENDATIONS
    # -----------------------------------------------------

    st.markdown("### 6. Quality Recommendations")

    recommendations = []

    if missing_cells > 0:

        recommendations.append(
            "Review missing values and decide whether to fill or remove them."
        )

    if duplicate_rows > 0:

        recommendations.append(
            "Review duplicate records before removing repeated rows."
        )

    if constant_columns:

        recommendations.append(
            "Review constant columns and consider removing unnecessary fields."
        )

    if not high_cardinality.empty:

        recommendations.append(
            "Review high-cardinality columns to identify unnecessary identifiers."
        )

    if not recommendations:

        st.success(
            "No basic quality issues were detected. "
            "Your dataset looks good based on these checks."
        )

    else:

        for index, recommendation in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"**{index}.** {recommendation}"
            )

    # -----------------------------------------------------
    # DOWNLOAD QUALITY REPORT
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Download Quality Report")

    report = dtype_report.merge(
        missing_report,
        on="Column",
        how="left"
    ).merge(
        cardinality_report,
        on="Column",
        how="left"
    )

    report_csv = report.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Data Quality Report",
        data=report_csv,
        file_name="data_quality_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "DataFlow AI | Data Quality Assessment"
    )

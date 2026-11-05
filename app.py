import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path
from html import escape

from backend.file_handler import load_file
from backend.cleaner import clean_data
from backend.analyzer import (
    get_overview,
    get_column_report,
    get_descriptive_statistics,
    get_missing_report,
    get_correlation,
)
from backend.visualizer import (
    create_histogram,
    create_bar_chart,
    create_boxplot,
    create_correlation_heatmap,
    create_missing_chart,
)
from backend.exporter import (
    to_csv_bytes,
    to_excel_bytes,
    analysis_to_excel_bytes,
)

from backend.trend_analyzer import (
    detect_date_columns,
    analyze_trends,
    generate_trend_insights,
)

from backend.data_quality import analyze_data_quality

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataFlow AI | Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# EXTERNAL STYLESHEET
# --------------------------------------------------
css_file = Path(__file__).resolve().parent / "assets" / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "last_uploaded_signature" not in st.session_state:
    st.session_state.last_uploaded_name = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("<div class='brand'><span class='brand-mark'>◈</span><span><strong>DataFlow AI</strong><small>ANALYTICS STUDIO</small></span></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-label'>WORKSPACE</div>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Clean Data",
        "Data Analysis",
        "Data Quality",
        "Trend Analysis",
        "Export Results",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("<div class='sidebar-label'>DATA SOURCE</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel dataset.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.last_uploaded_name:
        try:
            loaded_df = load_file(uploaded_file)

            st.session_state.original_df = loaded_df.copy(deep=True)
            st.session_state.cleaned_df = loaded_df.copy(deep=True)
            st.session_state.cleaning_log = [
                "Dataset uploaded successfully."
            ]
            st.session_state.source_name = uploaded_file.name
            st.session_state.last_uploaded_name = uploaded_file.name

            st.sidebar.success("Dataset loaded!")

        except Exception as error:
            st.sidebar.error(str(error))

            st.session_state.original_df = None
            st.session_state.cleaned_df = None
            st.session_state.cleaning_log = []
            st.session_state.source_name = None
            st.session_state.last_uploaded_name = None


# --------------------------------------------------
# HELPER: DOWNLOAD ORIGINAL OR CLEANED DATA
# --------------------------------------------------

original_df = st.session_state.original_df
cleaned_df = st.session_state.cleaned_df

if original_df is None or cleaned_df is None:
    st.markdown("<div class='eyebrow'>YOUR DATA WORKSPACE</div>", unsafe_allow_html=True)
    st.title("DataFlow AI")
    st.caption("Upload a CSV or Excel file to clean, explore, visualize and export your data.")
    st.info("Choose a file using the uploader in the left sidebar to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 01 · Clean")
        st.write("Prepare messy data with configurable cleaning operations.")
    with col2:
        st.markdown("### 02 · Analyze")
        st.write("Explore column quality, descriptive statistics and relationships.")
    with col3:
        st.markdown("### 03 · Visualize")
        st.write("See plots and insights directly inside the workspace.")
    st.stop()

original_df = st.session_state.original_df
cleaned_df = st.session_state.cleaned_df

st.markdown("<div class='eyebrow'>DATA CLEANING & ANALYTICS PLATFORM</div>", unsafe_allow_html=True)
st.title("DataFlow AI")
st.caption(f"Current dataset: {st.session_state.source_name}")

if st.sidebar.button("↺ Reset to Original Data", use_container_width=True):
    st.session_state.cleaned_df = st.session_state.original_df.copy(deep=True)
    st.session_state.cleaning_log = ["Restored the original dataset."]
    st.rerun()

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Analytics Dashboard")
    st.write("An overview of your current dataset.")

    metrics = get_overview(cleaned_df)

    original_rows = len(original_df)
    current_rows = len(cleaned_df)
    original_missing = int(original_df.isna().sum().sum())
    current_missing = int(cleaned_df.isna().sum().sum())

    total_cells = cleaned_df.size
    quality_score = (
        max(0, 100 * (1 - current_missing / total_cells))
        if total_cells else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Rows", f"{current_rows:,}")
    col2.metric("Total Columns", f"{len(cleaned_df.columns):,}")
    col3.metric(
        "Missing Values",
        f"{current_missing:,}",
        delta=current_missing - original_missing,
        delta_color="inverse",
    )
    col4.metric("Data Quality", f"{quality_score:.1f}%")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Duplicate Rows", f"{metrics['Duplicate Rows']:,}")
    col6.metric("Numerical Columns", f"{metrics['Numerical Columns']:,}")
    col7.metric("Categorical Columns", f"{metrics['Categorical Columns']:,}")
    col8.metric("Rows Removed", f"{max(0, original_rows-current_rows):,}")

    st.divider()

    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Number of rows to preview",
        min_value=5,
        max_value=50,
        value=10,
        key="dashboard_preview_rows",
    )

    st.dataframe(
        cleaned_df.head(preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Data Quality Overview")

    left, right = st.columns(2)

    with left:
        st.pyplot(create_missing_chart(cleaned_df), clear_figure=True)

    with right:
        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) >= 2:
            st.pyplot(
                create_correlation_heatmap(cleaned_df),
                clear_figure=True,
            )
        else:
            st.info(
                "Upload or select a dataset with at least two numerical "
                "columns to display a correlation heatmap."
            )

    st.subheader("Recent Cleaning Activity")

    if st.session_state.cleaning_log:
        for entry in reversed(st.session_state.cleaning_log[-10:]):
            st.write(f"• {entry}")
    else:
        st.caption("No cleaning activity yet.")


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

elif page == "Clean Data":

    st.header("Data Cleaning Studio")

    st.write(
        "Choose your cleaning operations, apply them, "
        "and inspect the results below."
    )

    tab1, tab2 = st.tabs(
        ["Original Data", "Cleaned Data"]
    )

    with tab1:
        st.subheader("Original Dataset")

        st.caption(
            f"{len(original_df):,} rows × {len(original_df.columns):,} columns"
        )

        st.dataframe(
            original_df,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.subheader("Current Cleaned Dataset")

        st.caption(
            f"{len(cleaned_df):,} rows × {len(cleaned_df.columns):,} columns"
        )

        st.dataframe(
            cleaned_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    if st.session_state.pop("cleaning_success", False):
        st.success("Cleaning completed. Your cleaned data and analysis are now updated.")

    st.subheader("Cleaning Options")

    with st.form("cleaning_form"):

        col1, col2 = st.columns(2)

        with col1:
            strip_spaces = st.checkbox(
                "Trim spaces and normalize blank text",
                value=True,
            )

            standardize_columns = st.checkbox(
                "Standardize column names",
                value=True,
            )

            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=True,
            )

        with col2:
            drop_empty_columns = st.checkbox(
                "Remove completely empty columns",
                value=False,
            )

            missing_strategy = st.selectbox(
                "Missing-value strategy",
                [
                    "Keep missing values",
                    "Drop rows with missing values",
                    "Fill numerical with median, text with mode",
                    "Fill numerical with mean, text with mode",
                ],
            )

        apply_cleaning = st.form_submit_button(
            "Apply Cleaning",
            type="primary",
            use_container_width=True,
        )

    if apply_cleaning:
        try:
            new_df, log = clean_data(
                st.session_state.cleaned_df,
                strip_spaces=strip_spaces,
                standardize_columns=standardize_columns,
                remove_duplicates=remove_duplicates,
                missing_strategy=missing_strategy,
                drop_empty_columns=drop_empty_columns,
            )

            st.session_state.cleaned_df = new_df
            st.session_state.cleaning_log.extend(log)

            st.session_state.cleaning_success = True
            st.rerun()

        except Exception as error:
            st.error(f"Cleaning failed: {error}")

    st.subheader("Before vs After")

    before, after = st.columns(2)

    with before:
        st.markdown("**Before cleaning**")
        st.metric("Rows", len(original_df))
        st.metric("Missing values", int(original_df.isna().sum().sum()))
        st.metric("Duplicate rows", int(original_df.duplicated().sum()))

    with after:
        st.markdown("**Current cleaned data**")
        st.metric("Rows", len(cleaned_df))
        st.metric("Missing values", int(cleaned_df.isna().sum().sum()))
        st.metric("Duplicate rows", int(cleaned_df.duplicated().sum()))

    st.subheader("Cleaning Log")

    if st.session_state.cleaning_log:
        st.dataframe(
            pd.DataFrame({
                "Cleaning Activity": st.session_state.cleaning_log
            }),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# DATA ANALYSIS
# --------------------------------------------------

elif page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    st.write(
        "Explore descriptive statistics, column quality, "
        "missing values, correlations and visualizations."
    )

    overview = get_overview(cleaned_df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{overview['Rows']:,}")
    col2.metric("Columns", f"{overview['Columns']:,}")
    col3.metric("Missing Values", f"{overview['Missing Values']:,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Column Report",
        "Statistics",
        "Missing Values",
        "Correlation",
        "Visualizations",
    ])

    with tab1:
        st.subheader("Column Quality Report")

        st.dataframe(
            get_column_report(cleaned_df),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.subheader("Descriptive Statistics")

        st.dataframe(
            get_descriptive_statistics(cleaned_df),
            use_container_width=True,
            hide_index=True,
        )

        categorical_columns = cleaned_df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if categorical_columns:
            st.subheader("Categorical Value Counts")

            selected_category = st.selectbox(
                "Select a categorical column",
                categorical_columns,
                key="category_analysis",
            )

            counts = (
                cleaned_df[selected_category]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Count")
            )

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        st.subheader("Missing Value Report")

        missing_report = get_missing_report(cleaned_df)

        st.dataframe(
            missing_report,
            use_container_width=True,
            hide_index=True,
        )

        st.pyplot(
            create_missing_chart(cleaned_df),
            clear_figure=True,
        )

    with tab4:
        st.subheader("Numerical Correlations")

        correlation = get_correlation(cleaned_df)

        if correlation.empty:
            st.info(
                "At least two numerical columns are needed for correlation analysis."
            )
        else:
            st.dataframe(correlation, use_container_width=True)

            st.pyplot(
                create_correlation_heatmap(cleaned_df),
                clear_figure=True,
            )

    with tab5:
        st.subheader("Create a Visualization")

        all_columns = cleaned_df.columns.tolist()

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Bar Chart",
                "Boxplot",
                "Correlation Heatmap",
            ],
        )

        if chart_type in ["Histogram", "Boxplot"]:
            numeric_columns = cleaned_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                selected_column = st.selectbox(
                    "Select a numerical column",
                    numeric_columns,
                )

                if chart_type == "Histogram":
                    fig = create_histogram(cleaned_df, selected_column)
                else:
                    fig = create_boxplot(cleaned_df, selected_column)

                st.pyplot(fig, clear_figure=True)

            else:
                st.warning("This chart requires at least one numerical column.")

        elif chart_type == "Bar Chart":
            if all_columns:
                selected_column = st.selectbox(
                    "Select a column",
                    all_columns,
                )

                top_n = st.slider(
                    "Number of categories",
                    min_value=3,
                    max_value=20,
                    value=10,
                )

                st.pyplot(
                    create_bar_chart(cleaned_df, selected_column, top_n),
                    clear_figure=True,
                )

        elif chart_type == "Correlation Heatmap":
            if len(cleaned_df.select_dtypes(include="number").columns) >= 2:
                st.pyplot(
                    create_correlation_heatmap(cleaned_df),
                    clear_figure=True,
                )
            else:
                st.warning(
                    "At least two numerical columns are required."
                )



# --------------------------------------------------
# SMART DATA QUALITY
# --------------------------------------------------

elif page == "Data Quality":

    st.header("Smart Data Quality")
    st.write(
        "Profile your dataset, identify data-quality issues, "
        "and review column-level quality scores."
    )

    quality_result = analyze_data_quality(cleaned_df)
    quality_score = quality_result["overall_score"]
    quality_grade = quality_result["grade"]
    quality_summary = quality_result["summary"]
    quality_report = quality_result["column_report"]
    quality_issues = pd.DataFrame(quality_result["issues"])

    # Overall score and grade
    score_col, grade_col, note_col = st.columns([1, 1, 2])

    with score_col:
        st.metric("Overall Quality Score", f"{quality_score:.1f}%")
        st.progress(int(round(quality_score)))

    with grade_col:
        st.metric("Quality Grade", quality_grade)

    with note_col:
        st.info(
            "This score is a rule-based indicator based on completeness, "
            "duplicate rows, formatting, and mixed data types. Review the "
            "detected issues before making data-cleaning decisions."
        )

    st.divider()
    st.subheader("Dataset Quality Overview")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing Cells", f"{quality_summary['Missing Cells']:,}",
              f"{quality_summary['Missing Percentage']:.1f}%")
    q2.metric("Duplicate Rows", f"{quality_summary['Duplicate Rows']:,}",
              f"{quality_summary['Duplicate Percentage']:.1f}%")
    q3.metric("Columns With Issues", f"{quality_summary['Columns With Issues']:,}")
    q4.metric("Mixed-Type Columns", f"{quality_summary['Columns With Mixed Types']:,}")

    st.caption(
        f"Analyzing the current dataset: {len(cleaned_df):,} rows × "
        f"{len(cleaned_df.columns):,} columns."
    )

    st.divider()
    st.subheader("Column Quality Report")

    if quality_report.empty:
        st.info("No columns are available for quality analysis.")
    else:
        st.dataframe(
            quality_report,
            use_container_width=True,
            hide_index=True,
        )

        # Column quality score chart
        score_chart_data = quality_report.sort_values(
            "Quality Score", ascending=True
        )
        fig, ax = plt.subplots(figsize=(10, max(3, len(score_chart_data) * 0.45)))
        bars = ax.barh(
            score_chart_data["Column"].astype(str),
            score_chart_data["Quality Score"],
            color="#0f9f93",
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Quality Score (%)")
        ax.set_title("Quality Score by Column", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.divider()
    st.subheader("Detected Data Quality Issues")

    if quality_issues.empty:
        st.success("No issues were detected in the current dataset.")
    else:
        issue_counts = (
            quality_issues["issue"]
            .value_counts()
            .rename_axis("Issue")
            .reset_index(name="Affected Columns")
        )
        issue_col, details_col = st.columns([1, 2])

        with issue_col:
            st.markdown("**Issue distribution**")
            st.dataframe(
                issue_counts,
                use_container_width=True,
                hide_index=True,
            )

        with details_col:
            st.markdown("**Issue details**")
            st.dataframe(
                quality_issues,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "These are detected patterns, not automatic corrections. "
            "Review each issue before changing your dataset."
        )

    st.divider()
    st.subheader("Download Quality Results")

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download Column Quality Report (CSV)",
            data=quality_report.to_csv(index=False).encode("utf-8-sig"),
            file_name="data_quality_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        st.download_button(
            "Download Detected Issues (CSV)",
            data=quality_issues.to_csv(index=False).encode("utf-8-sig"),
            file_name="data_quality_issues.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------------------------
# TRENDS ANALYSIS
# --------------------------------------------------


# ==========================================
# ADVANCED TREND ANALYSIS
# ==========================================

elif page == "Trend Analysis":

    st.header("📈 Advanced Trend Analysis")

    st.write(
        "Explore trends, growth rates, moving averages, "
        "and changes in your dataset over time."
    )

    date_columns = detect_date_columns(cleaned_df)

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not date_columns:
        st.warning(
            "No date columns detected. "
            "Your dataset must contain valid dates."
        )

    elif not numeric_columns:
        st.warning(
            "Trend analysis requires at least one numerical column."
        )

    else:

        # ======================================
        # CONFIGURATION
        # ======================================

        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)

        with col1:
            selected_date_column = st.selectbox(
                "Select Date Column",
                date_columns,
            )

            selected_value_column = st.selectbox(
                "Select Numerical Column",
                numeric_columns,
            )

        with col2:
            frequency = st.selectbox(
                "Analysis Frequency",
                ["Daily", "Weekly", "Monthly", "Yearly"],
                index=0,
            )

            aggregation = st.selectbox(
                "Aggregation Method",
                ["Sum", "Average", "Count", "Minimum", "Maximum"],
            )

        try:

            result, summary = analyze_trends(
                cleaned_df,
                selected_date_column,
                selected_value_column,
                frequency,
                aggregation,
            )

            # ======================================
            # KPI CARDS
            # ======================================

            st.divider()
            st.subheader("Trend Overview")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Records Analyzed",
                f"{summary['Total Records']:,}",
            )

            c2.metric(
                "Time Periods",
                f"{summary['Periods']:,}",
            )

            c3.metric(
                "Latest Period Value",
                f"{summary['Latest Period Value']:,.2f}",
            )

            growth = summary["Latest Growth %"]

            c4.metric(
                "Latest Growth",
                f"{growth:+.2f}%" if growth is not None else "N/A",
            )

            # ======================================
            # TREND VISUALIZATION
            # ======================================

            st.divider()
            st.subheader("Time Series Trend")

            if len(result) < 2:

                st.warning(
                    "Only one time period is available. "
                    "A trend requires at least two periods. "
                    "Try selecting Daily frequency or upload "
                    "a dataset covering a longer time range."
                )

            fig, ax = plt.subplots(figsize=(12, 5))

            if len(result) == 1:

                ax.scatter(
                    result["Date"],
                    result["Value"],
                    color="#0f9f93",
                    s=110,
                    zorder=3,
                    label="Observed Value",
                )

                ax.annotate(
                    f"{result['Value'].iloc[0]:,.2f}",
                    (
                        result["Date"].iloc[0],
                        result["Value"].iloc[0],
                    ),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=11,
                )

                ax.set_xlim(
                    result["Date"].iloc[0] - pd.Timedelta(days=2),
                    result["Date"].iloc[0] + pd.Timedelta(days=2),
                )

            else:

                ax.plot(
                    result["Date"],
                    result["Value"],
                    color="#0f9f93",
                    linewidth=2.5,
                    marker="o",
                    markersize=5,
                    label="Actual Value",
                )

                ax.plot(
                    result["Date"],
                    result["Moving Average"],
                    color="#14233f",
                    linewidth=2,
                    linestyle="--",
                    label="3-Period Moving Average",
                )

                ax.margins(x=0.03)

            ax.set_title(
                f"{frequency} {aggregation} Trend",
                fontsize=15,
                fontweight="bold",
            )

            ax.set_xlabel("Date")
            ax.set_ylabel(selected_value_column)
            ax.grid(alpha=0.2)
            ax.legend()

            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            # ======================================
            # PERIOD COMPARISON TABLE
            # ======================================

            st.divider()
            st.subheader("Period-over-Period Analysis")

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Growth compares consecutive observed periods. "
                "The moving average uses the latest three "
                "observed periods."
            )

            # ======================================
            # AUTOMATED INSIGHTS
            # ======================================

            st.divider()
            st.subheader("Automated Trend Insights")

            insights = generate_trend_insights(
                result,
                summary,
            )

            for insight in insights:
                st.info(insight)

            # ======================================
            # EXPORT
            # ======================================

            st.divider()
            st.subheader("Export Trend Results")

            csv_data = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Trend Analysis CSV",
                data=csv_data,
                file_name="trend_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except (ValueError, KeyError, TypeError) as error:
            st.error(
                f"Trend analysis failed: {error}"
            )


# --------------------------------------------------
# EXPORT RESULTS
# --------------------------------------------------

elif page == "Export Results":

    st.header("Export Results")

    st.write(
        "Download your cleaned dataset or export analysis tables "
        "for further work."
    )

    st.subheader("Cleaned Dataset")

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Cleaned CSV",
            data=to_csv_bytes(cleaned_df),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Download Cleaned Excel",
            data=to_excel_bytes(cleaned_df),
            file_name="cleaned_data.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Export Analysis Tables")

    analysis_tables = {
        "Column Report": get_column_report(cleaned_df),
        "Descriptive Stats": get_descriptive_statistics(cleaned_df),
        "Missing Values": get_missing_report(cleaned_df),
    }

    correlation = get_correlation(cleaned_df)

    if not correlation.empty:
        analysis_tables["Correlation"] = correlation.reset_index()

    for name, table in analysis_tables.items():
        with st.expander(name):
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

    st.download_button(
        "Download All Analysis Tables (Excel)",
        data=analysis_to_excel_bytes(analysis_tables),
        file_name="data_analysis_report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("DataFlow AI · Python Analytics Studio")
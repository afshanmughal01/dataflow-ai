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
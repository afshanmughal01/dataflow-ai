
import streamlit as st
import pandas as pd

from pathlib import Path

from backend.file_handler import load_file

from app_pages.dashboard import render_dashboard
from app_pages.data_preparation import render_data_preparation
from app_pages.data_analysis import render_data_analysis
from app_pages.data_quality import render_data_quality
from app_pages.trend_analysis import render_trend_analysis
from app_pages.forecasting import render_forecasting
from app_pages.business_insights import render_business_insights
from app_pages.export_center import render_export_center

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DataFlow AI | Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD EXISTING THEME
# =========================================================

css_file = (
    Path(__file__).resolve().parent
    / "assets"
    / "style.css"
)

if css_file.exists():

    st.markdown(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "original_df": None,
    "cleaned_df": None,
    "prepared_df": None,
    "cleaning_log": [],
    "source_name": None,
    "last_uploaded_name": None
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR BRANDING
# =========================================================

st.sidebar.markdown(
    """
    <div class="brand">
        <span class="brand-mark">◈</span>
        <span>
            <strong>DataFlow AI</strong>
            <small>ANALYTICS STUDIO</small>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown(
    "<div class='sidebar-label'>OVERVIEW</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Data Preparation",
        "Data Analysis",
        "Data Quality",
        "Trend Analysis",
        "Forecasting & Prediction",
        "Business Insights",
        "Export Center"
    ],
    label_visibility="collapsed"
)


# =========================================================
# DATA UPLOAD
# =========================================================

st.sidebar.divider()

st.sidebar.markdown(
    "<div class='sidebar-label'>DATA SOURCE</div>",
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel dataset."
)


if uploaded_file is not None:

    if uploaded_file.name != st.session_state.last_uploaded_name:

        try:

            loaded_df = load_file(uploaded_file)

            if loaded_df is None or loaded_df.empty:
                st.sidebar.error("The uploaded dataset is empty.")

            else:

                st.session_state.original_df = (
                    loaded_df.copy(deep=True)
                )

                st.session_state.cleaned_df = (
                    loaded_df.copy(deep=True)
                )

                st.session_state.prepared_df = None

                st.session_state.cleaning_log = [
                    "Dataset uploaded successfully."
                ]

                st.session_state.source_name = uploaded_file.name

                st.session_state.last_uploaded_name = (
                    uploaded_file.name
                )

                st.sidebar.success("Dataset loaded successfully!")

        except Exception as error:

            st.sidebar.error(
                f"Unable to load dataset: {error}"
            )

            st.session_state.original_df = None
            st.session_state.cleaned_df = None
            st.session_state.prepared_df = None
            st.session_state.cleaning_log = []
            st.session_state.source_name = None
            st.session_state.last_uploaded_name = None


# =========================================================
# DATASET STATUS
# =========================================================

original_df = st.session_state.original_df

if original_df is None:

    st.markdown(
        "<div class='eyebrow'>YOUR DATA WORKSPACE</div>",
        unsafe_allow_html=True
    )

    st.title("DataFlow AI")

    st.caption(
        "Your workspace for data preparation, analytics, "
        "forecasting and business intelligence."
    )

    st.info(
        "Upload a CSV or Excel file using the left sidebar "
        "to start working with your data."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 01 · Prepare")

        st.write(
            "Clean missing values, duplicates and inconsistent data."
        )

    with col2:

        st.markdown("### 02 · Analyze")

        st.write(
            "Explore statistics, data quality and trends."
        )

    with col3:

        st.markdown("### 03 · Predict")

        st.write(
            "Forecast future values and generate business insights."
        )

    st.stop()


# =========================================================
# ACTIVE DATASET
# =========================================================

if isinstance(st.session_state.prepared_df, pd.DataFrame):

    active_df = st.session_state.prepared_df

else:

    active_df = st.session_state.cleaned_df

if active_df is None:

    active_df = original_df.copy(deep=True)


# =========================================================
# WORKSPACE HEADER
# =========================================================

st.markdown(
    "<div class='eyebrow'>DATA CLEANING & ANALYTICS PLATFORM</div>",
    unsafe_allow_html=True
)

st.title("DataFlow AI")

st.caption(
    f"Current dataset: {st.session_state.source_name}"
)

st.sidebar.caption(
    f"Dataset: {st.session_state.source_name}"
)

st.sidebar.caption(
    f"Rows: {len(active_df):,} | Columns: {len(active_df.columns):,}"
)


# =========================================================
# RESET DATA
# =========================================================

st.sidebar.divider()

if st.sidebar.button(
    "↺ Reset to Original Data",
    use_container_width=True
):

    st.session_state.cleaned_df = (
        st.session_state.original_df.copy(deep=True)
    )

    st.session_state.prepared_df = None

    st.session_state.cleaning_log = [
        "Dataset restored to its original state."
    ]

    st.rerun()


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "Dashboard":

    render_dashboard(active_df)


elif page == "Data Preparation":

    result = render_data_preparation(active_df)

    if isinstance(result, tuple) and len(result) == 2:

        prepared_df, cleaning_log = result

        if isinstance(prepared_df, pd.DataFrame):

            st.session_state.prepared_df = (
                prepared_df.copy(deep=True)
            )

            st.session_state.cleaned_df = (
                prepared_df.copy(deep=True)
            )

        if cleaning_log is not None:

            st.session_state.cleaning_log = cleaning_log


elif page == "Data Analysis":

    render_data_analysis(active_df)


elif page == "Data Quality":

    render_data_quality(active_df)


elif page == "Trend Analysis":

    render_trend_analysis(active_df)


elif page == "Forecasting & Prediction":

    render_forecasting(active_df)


elif page == "Business Insights":

    render_business_insights(active_df)


elif page == "Export Center":

    render_export_center(active_df)


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "DataFlow AI | Python Analytics Studio"
)

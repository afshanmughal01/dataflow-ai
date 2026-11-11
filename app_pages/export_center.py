
import streamlit as st
import pandas as pd
import numpy as np
import io

from datetime import datetime


# =========================================================
# EXPORT CENTER CONFIGURATION
# =========================================================

APP_NAME = "DataFlow AI"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_bytes(size):
    """Convert bytes into a readable file size."""

    size = float(size)

    for unit in ["B", "KB", "MB", "GB"]:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} TB"


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


def prepare_export_data(df, include_index):
    """Prepare the dataset consistently for different export formats."""

    export_data = df.copy(deep=True)

    if include_index:

        index_name = export_data.index.name or "Index"

        # Avoid overwriting an existing column
        if index_name in export_data.columns:

            index_name = "Export_Index"

            counter = 1

            while index_name in export_data.columns:

                index_name = f"Export_Index_{counter}"
                counter += 1

        export_data.insert(
            0,
            index_name,
            export_data.index
        )

    return export_data


def generate_summary_report(df, timestamp):
    """Generate a detailed dataset summary report."""

    numeric_columns = get_numeric_columns(df)

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    memory_usage = df.memory_usage(
        index=True,
        deep=True
    ).sum()

    summary_lines = [
        f"{APP_NAME} - DATASET SUMMARY REPORT",
        "=" * 55,
        "",
        f"Generated: {timestamp}",
        "",
        "DATASET OVERVIEW",
        "-" * 55,
        f"Total Rows: {len(df):,}",
        f"Total Columns: {len(df.columns):,}",
        f"Missing Cells: {missing_cells:,}",
        f"Duplicate Rows: {duplicate_rows:,}",
        f"Memory Usage: {format_bytes(memory_usage)}",
        "",
        "COLUMN INFORMATION",
        "-" * 55
    ]

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        summary_lines.append(
            f"{column}: "
            f"Type={df[column].dtype}, "
            f"Missing={missing_count:,} "
            f"({missing_percentage:.2f}%), "
            f"Unique={df[column].nunique(dropna=True):,}"
        )

    if numeric_columns:

        numeric_data = pd.DataFrame(index=df.index)

        for column in numeric_columns:

            numeric_data[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        summary_lines.extend([
            "",
            "NUMERICAL SUMMARY",
            "-" * 55,
            numeric_data.describe().round(2).to_string()
        ])

    return "\n".join(summary_lines)


def generate_cleaning_log(cleaning_log):
    """Convert the cleaning history into downloadable text."""

    if isinstance(cleaning_log, list):

        return "\n".join(
            str(item) for item in cleaning_log
        )

    if isinstance(cleaning_log, dict):

        return "\n".join(
            f"{key}: {value}"
            for key, value in cleaning_log.items()
        )

    return str(cleaning_log)


# =========================================================
# MAIN EXPORT CENTER PAGE
# =========================================================

def render_export_center(df):

    st.markdown(
        '<p class="eyebrow">REPORTING & DATA EXPORT</p>',
        unsafe_allow_html=True
    )

    st.title("Export Center")

    st.markdown(
        "Download your datasets and analytical reports "
        "in convenient formats."
    )

    # -----------------------------------------------------
    # VALIDATE DATA
    # -----------------------------------------------------

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:

        st.warning(
            "Please upload a dataset first."
        )

        return

    # -----------------------------------------------------
    # SELECT DATASET
    # -----------------------------------------------------

    original_df = st.session_state.get(
        "original_df"
    )

    prepared_df = st.session_state.get(
        "prepared_df"
    )

    dataset_options = [
        "Current Dataset"
    ]

    if (
        isinstance(original_df, pd.DataFrame)
        and not original_df.empty
    ):

        dataset_options.append(
            "Original Dataset"
        )

    if (
        isinstance(prepared_df, pd.DataFrame)
        and not prepared_df.empty
    ):

        dataset_options.append(
            "Prepared Dataset"
        )

    st.markdown("### Select Dataset")

    selected_dataset = st.selectbox(
        "Choose the dataset you want to export",
        dataset_options,
        key="export_dataset_selection"
    )

    if selected_dataset == "Original Dataset":

        export_df = original_df.copy(deep=True)

    elif selected_dataset == "Prepared Dataset":

        export_df = prepared_df.copy(deep=True)

    else:

        export_df = df.copy(deep=True)

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Dataset Information")

    missing_cells = int(
        export_df.isna().sum().sum()
    )

    duplicate_rows = int(
        export_df.duplicated().sum()
    )

    memory_usage = export_df.memory_usage(
        index=True,
        deep=True
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        f"{len(export_df):,}"
    )

    col2.metric(
        "Total Columns",
        f"{len(export_df.columns):,}"
    )

    col3.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )

    st.caption(
        f"Dataset memory usage: {format_bytes(memory_usage)}"
    )

    # -----------------------------------------------------
    # DATASET PREVIEW
    # -----------------------------------------------------

    st.markdown("### Dataset Preview")

    st.dataframe(
        export_df.head(10),
        width="stretch",
        hide_index=True
    )

    # -----------------------------------------------------
    # EXPORT CONFIGURATION
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Export Configuration")

    col1, col2 = st.columns(2)

    with col1:

        export_format = st.selectbox(
            "Select Export Format",
            [
                "CSV",
                "Excel",
                "JSON",
                "TXT"
            ],
            key="export_format"
        )

    with col2:

        include_index = st.checkbox(
            "Include DataFrame Index",
            value=False,
            key="export_include_index"
        )

    # -----------------------------------------------------
    # PREPARE EXPORT DATA
    # -----------------------------------------------------

    file_export_df = prepare_export_data(
        export_df,
        include_index
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    file_data = None
    file_name = None
    mime_type = None

    export_error = None

    # -----------------------------------------------------
    # GENERATE CSV
    # -----------------------------------------------------

    try:

        if export_format == "CSV":

            file_data = file_export_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            file_name = (
                f"dataflow_export_{timestamp}.csv"
            )

            mime_type = "text/csv"

        # -------------------------------------------------
        # GENERATE EXCEL
        # -------------------------------------------------

        elif export_format == "Excel":

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                file_export_df.to_excel(
                    writer,
                    sheet_name="Dataset",
                    index=False
                )

                # Add a separate dataset information sheet
                info_df = pd.DataFrame({
                    "Information": [
                        "Total Rows",
                        "Total Columns",
                        "Missing Cells",
                        "Duplicate Rows",
                        "Memory Usage"
                    ],
                    "Value": [
                        len(export_df),
                        len(export_df.columns),
                        missing_cells,
                        duplicate_rows,
                        format_bytes(memory_usage)
                    ]
                })

                info_df.to_excel(
                    writer,
                    sheet_name="Dataset Info",
                    index=False
                )

            file_data = excel_buffer.getvalue()

            file_name = (
                f"dataflow_export_{timestamp}.xlsx"
            )

            mime_type = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )

        # -------------------------------------------------
        # GENERATE JSON
        # -------------------------------------------------

        elif export_format == "JSON":

            file_data = file_export_df.to_json(
                orient="records",
                indent=4,
                date_format="iso",
                force_ascii=False
            ).encode("utf-8")

            file_name = (
                f"dataflow_export_{timestamp}.json"
            )

            mime_type = "application/json"

        # -------------------------------------------------
        # GENERATE TXT
        # -------------------------------------------------

        elif export_format == "TXT":

            file_data = file_export_df.to_string(
                index=False
            ).encode("utf-8")

            file_name = (
                f"dataflow_export_{timestamp}.txt"
            )

            mime_type = "text/plain"

    except Exception as error:

        export_error = str(error)

    # -----------------------------------------------------
    # DOWNLOAD DATASET
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Download Dataset")

    if export_error:

        st.error(
            f"Unable to generate the export file: {export_error}"
        )

    elif file_data is not None:

        st.caption(
            f"File: {file_name} | "
            f"Size: {format_bytes(len(file_data))}"
        )

        st.download_button(
            label=f"Download {export_format} File",
            data=file_data,
            file_name=file_name,
            mime=mime_type,
            width="stretch",
            key="download_dataset"
        )

    # -----------------------------------------------------
    # GENERATE DATA SUMMARY
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Dataset Summary Report")

    summary_text = generate_summary_report(
        export_df,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    st.text_area(
        "Report Preview",
        value=summary_text,
        height=300,
        key="export_summary_preview"
    )

    st.download_button(
        label="Download Summary Report",
        data=summary_text.encode("utf-8-sig"),
        file_name=f"dataflow_summary_{timestamp}.txt",
        mime="text/plain",
        width="stretch",
        key="download_summary"
    )

    # -----------------------------------------------------
    # CLEANING LOG EXPORT
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Cleaning History")

    cleaning_log = st.session_state.get(
        "cleaning_log",
        []
    )

    if cleaning_log:

        log_text = generate_cleaning_log(
            cleaning_log
        )

        if log_text.strip():

            st.text_area(
                "Cleaning Log Preview",
                value=log_text,
                height=180,
                key="export_cleaning_log_preview"
            )

            st.download_button(
                label="Download Cleaning Log",
                data=log_text.encode("utf-8-sig"),
                file_name=f"dataflow_cleaning_log_{timestamp}.txt",
                mime="text/plain",
                width="stretch",
                key="download_cleaning_log"
            )

        else:

            st.info(
                "The cleaning history is currently empty."
            )

    else:

        st.info(
            "No cleaning history is available yet."
        )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()

    st.caption(
        "DataFlow AI | Export Center"
    )

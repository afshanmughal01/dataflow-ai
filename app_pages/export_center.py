
import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime


# =========================================================
# EXPORT CENTER
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

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
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

    dataset_options = ["Current Dataset"]

    if isinstance(original_df, pd.DataFrame) and not original_df.empty:
        dataset_options.append("Original Dataset")

    if isinstance(prepared_df, pd.DataFrame) and not prepared_df.empty:
        dataset_options.append("Prepared Dataset")

    st.markdown("### Select Dataset")

    selected_dataset = st.selectbox(
        "Choose the dataset you want to export",
        dataset_options
    )

    if selected_dataset == "Original Dataset":
        export_df = original_df.copy()

    elif selected_dataset == "Prepared Dataset":
        export_df = prepared_df.copy()

    else:
        export_df = df.copy()

    # -----------------------------------------------------
    # DATASET INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Dataset Information")

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
        f"{export_df.isna().sum().sum():,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{export_df.duplicated().sum():,}"
    )

    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    st.markdown("### Dataset Preview")

    st.dataframe(
        export_df.head(10),
        use_container_width=True,
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
            ]
        )

    with col2:

        include_index = st.checkbox(
            "Include DataFrame Index",
            value=False
        )

    # -----------------------------------------------------
    # GENERATE FILE
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    file_data = None
    file_name = None
    mime_type = None

    if export_format == "CSV":

        file_data = export_df.to_csv(
            index=include_index
        ).encode("utf-8-sig")

        file_name = f"dataflow_export_{timestamp}.csv"

        mime_type = "text/csv"

    elif export_format == "Excel":

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            export_df.to_excel(
                writer,
                sheet_name="Dataset",
                index=include_index
            )

        file_data = excel_buffer.getvalue()

        file_name = f"dataflow_export_{timestamp}.xlsx"

        mime_type = (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )

    elif export_format == "JSON":

        file_data = export_df.to_json(
            orient="records",
            indent=4,
            date_format="iso"
        ).encode("utf-8")

        file_name = f"dataflow_export_{timestamp}.json"

        mime_type = "application/json"

    elif export_format == "TXT":

        file_data = export_df.to_string(
            index=include_index
        ).encode("utf-8")

        file_name = f"dataflow_export_{timestamp}.txt"

        mime_type = "text/plain"

    # -----------------------------------------------------
    # DOWNLOAD DATASET
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Download Dataset")

    st.download_button(
        label=f"Download {export_format} File",
        data=file_data,
        file_name=file_name,
        mime=mime_type,
        use_container_width=True
    )

    # -----------------------------------------------------
    # GENERATE DATA SUMMARY
    # -----------------------------------------------------

    st.divider()

    st.markdown("### Dataset Summary Report")

    numeric_columns = export_df.select_dtypes(
        include="number"
    ).columns.tolist()

    summary_lines = [
        "DATAFLOW AI - DATASET SUMMARY REPORT",
        "=" * 45,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Rows: {len(export_df)}",
        f"Total Columns: {len(export_df.columns)}",
        f"Missing Cells: {export_df.isna().sum().sum()}",
        f"Duplicate Rows: {export_df.duplicated().sum()}",
        "",
        "COLUMN INFORMATION",
        "-" * 45
    ]

    for column in export_df.columns:

        summary_lines.append(
            f"{column}: {export_df[column].dtype}"
        )

    if numeric_columns:

        summary_lines.extend([
            "",
            "NUMERICAL SUMMARY",
            "-" * 45,
            export_df[numeric_columns].describe().round(2).to_string()
        ])

    summary_text = "\n".join(summary_lines)

    st.text_area(
        "Report Preview",
        value=summary_text,
        height=300
    )

    st.download_button(
        label="Download Summary Report",
        data=summary_text.encode("utf-8"),
        file_name=f"dataflow_summary_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True
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

        if isinstance(cleaning_log, list):

            log_text = "\n".join(
                str(item) for item in cleaning_log
            )

        else:

            log_text = str(cleaning_log)

        st.text_area(
            "Cleaning Log Preview",
            value=log_text,
            height=180
        )

        st.download_button(
            label="Download Cleaning Log",
            data=log_text.encode("utf-8"),
            file_name=f"dataflow_cleaning_log_{timestamp}.txt",
            mime="text/plain",
            use_container_width=True
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

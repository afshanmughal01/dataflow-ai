import io
import pandas as pd


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")

    output.seek(0)
    return output.getvalue()


def analysis_to_excel_bytes(tables):
    """Export multiple analysis tables into one Excel workbook."""

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, index=False, sheet_name=safe_name)

    output.seek(0)
    return output.getvalue()
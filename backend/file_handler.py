import io
import pandas as pd


SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls"]


def load_file(uploaded_file):
    """Load and validate an uploaded CSV or Excel file."""

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            content = uploaded_file.getvalue()

            for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode this CSV file.")

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError("Please upload a CSV or Excel file.")

        if df.empty:
            raise ValueError("The uploaded file contains no data.")

        if len(df.columns) == 0:
            raise ValueError("The uploaded file has no columns.")

        return df

    except Exception as error:
        raise ValueError(f"Could not load the file: {error}") from error
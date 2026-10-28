import pandas as pd
import numpy as np


def get_overview(df):
    """Return the main dataset statistics."""

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Numerical Columns": len(df.select_dtypes(include="number").columns),
        "Categorical Columns": len(
            df.select_dtypes(include=["object", "category", "string"]).columns
        ),
    }


def get_column_report(df):
    """Report data types, missing values, uniqueness and memory."""

    report = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Count": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
        "Duplicate Values": [
            int(df[column].duplicated().sum()) for column in df.columns
        ],
    })

    return report


def get_descriptive_statistics(df):
    """Summarize numerical and categorical columns."""

    if df.empty:
        return pd.DataFrame()

    return df.describe(include="all").transpose().reset_index(names="Column")


def get_missing_report(df):
    """Return columns with missing values."""

    report = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Missing Count": df.isna().sum().values,
        "Missing Percentage": (df.isna().mean().values * 100).round(2),
    })

    return report.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def get_correlation(df):
    """Calculate correlations for numerical columns."""

    numerical = df.select_dtypes(include=np.number)

    if numerical.shape[1] < 2:
        return pd.DataFrame()

    return numerical.corr()
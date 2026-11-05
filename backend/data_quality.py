
import pandas as pd
import numpy as np


def analyze_data_quality(df):

    if df is None or df.empty:
        return {
            "overall_score": 0,
            "grade": "No Data",
            "summary": {},
            "column_report": pd.DataFrame(),
            "issues": []
        }

    total_rows = len(df)
    total_columns = len(df.columns)

    # --------------------------------
    # 1. Dataset-level metrics
    # --------------------------------

    missing_cells = int(df.isna().sum().sum())
    total_cells = total_rows * total_columns

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells else 0
    )

    duplicate_rows = int(df.duplicated().sum())

    duplicate_percentage = (
        duplicate_rows / total_rows * 100
        if total_rows else 0
    )

    # --------------------------------
    # 2. Column-level analysis
    # --------------------------------

    column_results = []
    issues = []

    for column in df.columns:

        series = df[column]

        missing_count = int(series.isna().sum())

        missing_pct = (
            missing_count / total_rows * 100
            if total_rows else 0
        )

        non_null = series.dropna()

        unique_count = int(non_null.nunique())

        # Detect extra spaces in text values
        whitespace_count = 0

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):

            text_values = non_null.astype(str)

            whitespace_count = int(
                text_values.apply(
                    lambda value: value != value.strip()
                ).sum()
            )

        # Detect possible mixed data types
        type_count = int(
            non_null.apply(lambda value: type(value).__name__).nunique()
        )

        mixed_types = type_count > 1

        # Detect numeric values stored as text
        numeric_as_text = False

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):

            if len(non_null) > 0:

                converted = pd.to_numeric(
                    non_null,
                    errors="coerce"
                )

                numeric_as_text = (
                    converted.notna().mean() >= 0.8
                )

        # Detect date values stored as text
        date_as_text = False

        if (
            not numeric_as_text
            and (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )
            and len(non_null) > 0
        ):

            date_sample = non_null.astype(str).head(100)

            date_converted = pd.to_datetime(
                date_sample,
                errors="coerce",
                format="mixed"
            )

            date_as_text = (
                date_converted.notna().mean() >= 0.8
                and date_sample.str.contains(
                    r"[-/]",
                    regex=True
                ).mean() >= 0.5
            )

        # --------------------------------
        # 3. Column quality score
        # --------------------------------

        completeness_score = 100 - missing_pct

        formatting_penalty = min(
            whitespace_count / max(total_rows, 1) * 100,
            100
        )

        type_penalty = 100 if mixed_types else 0

        column_score = (
            completeness_score * 0.50
            + (100 - formatting_penalty) * 0.25
            + (100 - type_penalty) * 0.25
        )

        column_score = round(
            max(0, min(column_score, 100)),
            2
        )

        # --------------------------------
        # 4. Identify issues
        # --------------------------------

        column_issues = []

        if missing_count > 0:
            column_issues.append("Missing Values")

        if whitespace_count > 0:
            column_issues.append("Extra Spaces")

        if mixed_types:
            column_issues.append("Mixed Data Types")

        if numeric_as_text:
            column_issues.append("Possible Numeric Data Stored as Text")

        if date_as_text:
            column_issues.append("Possible Date Stored as Text")

        for issue in column_issues:

            issues.append({
                "column": column,
                "issue": issue,
                "affected_records": (
                    missing_count if issue == "Missing Values"
                    else whitespace_count if issue == "Extra Spaces"
                    else None
                )
            })

        column_results.append({
            "Column": column,
            "Data Type": str(series.dtype),
            "Missing Values": missing_count,
            "Missing %": round(missing_pct, 2),
            "Unique Values": unique_count,
            "Extra Spaces": whitespace_count,
            "Mixed Types": "Yes" if mixed_types else "No",
            "Numeric as Text": "Yes" if numeric_as_text else "No",
            "Date as Text": "Yes" if date_as_text else "No",
            "Quality Score": column_score,
            "Issues": ", ".join(column_issues) if column_issues else "No Issues"
        })

    column_report = pd.DataFrame(column_results)

    # --------------------------------
    # 5. Overall dataset quality score
    # --------------------------------

    completeness_score = 100 - missing_percentage

    duplicate_score = 100 - duplicate_percentage

    total_whitespace = sum(
        result["Extra Spaces"]
        for result in column_results
    )

    formatting_percentage = (
        total_whitespace / total_cells * 100
        if total_cells else 0
    )

    formatting_score = 100 - formatting_percentage

    mixed_type_columns = sum(
        result["Mixed Types"] == "Yes"
        for result in column_results
    )

    type_score = (
        100 - (mixed_type_columns / total_columns * 100)
        if total_columns else 100
    )

    overall_score = (
        completeness_score * 0.40
        + duplicate_score * 0.20
        + formatting_score * 0.20
        + type_score * 0.20
    )

    overall_score = round(
        max(0, min(overall_score, 100)),
        2
    )

    if overall_score >= 90:
        grade = "Excellent"
    elif overall_score >= 75:
        grade = "Good"
    elif overall_score >= 60:
        grade = "Fair"
    else:
        grade = "Poor"

    # --------------------------------
    # 6. Summary
    # --------------------------------

    summary = {
        "Total Rows": total_rows,
        "Total Columns": total_columns,
        "Missing Cells": missing_cells,
        "Missing Percentage": round(missing_percentage, 2),
        "Duplicate Rows": duplicate_rows,
        "Duplicate Percentage": round(duplicate_percentage, 2),
        "Columns With Mixed Types": mixed_type_columns,
        "Columns With Issues": int(
            (column_report["Issues"] != "No Issues").sum()
        )
    }

    return {
        "overall_score": overall_score,
        "grade": grade,
        "summary": summary,
        "column_report": column_report,
        "issues": issues
    }

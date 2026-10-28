import pandas as pd
import numpy as np


def clean_data(
    df,
    strip_spaces=True,
    standardize_columns=True,
    remove_duplicates=True,
    missing_strategy="Keep missing values",
    drop_empty_columns=False,
):
    """Clean a copy of the dataset and return it with a change log."""

    if df is None or df.empty:
        raise ValueError("Please upload a non-empty dataset.")

    cleaned = df.copy(deep=True)
    log = []

    initial_rows = len(cleaned)
    initial_columns = len(cleaned.columns)
    initial_missing = int(cleaned.isna().sum().sum())

    # Standardize column names.
    if strip_spaces or standardize_columns:
        new_columns = []

        for column in cleaned.columns:
            name = str(column).strip() if strip_spaces else str(column)

            if standardize_columns:
                name = name.lower()
                name = "_".join(name.split())

            new_columns.append(name)

        # Ensure duplicate column names do not overwrite each other.
        unique_columns = []
        seen = {}

        for name in new_columns:
            seen[name] = seen.get(name, 0) + 1
            unique_columns.append(
                name if seen[name] == 1 else f"{name}_{seen[name]}"
            )

        if list(cleaned.columns) != unique_columns:
            cleaned.columns = unique_columns
            log.append("Standardized column names.")

    # Strip whitespace from text cells.
    if strip_spaces:
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

        log.append("Removed leading and trailing spaces from text values.")

    # Replace blank strings with missing values.
    if strip_spaces:
        for column in cleaned.select_dtypes(include=["object", "string"]).columns:
            cleaned[column] = cleaned[column].replace(r"^\s*$", np.nan, regex=True)

    # Drop completely empty columns.
    if drop_empty_columns:
        empty_columns = cleaned.columns[cleaned.isna().all()].tolist()
        cleaned = cleaned.drop(columns=empty_columns)

        if empty_columns:
            log.append(f"Removed {len(empty_columns)} completely empty columns.")

    # Remove duplicate rows.
    if remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        removed = before - len(cleaned)
        log.append(f"Removed {removed} duplicate rows.")

    # Handle missing values.
    missing_before = int(cleaned.isna().sum().sum())

    if missing_strategy == "Drop rows with missing values":
        before = len(cleaned)
        cleaned = cleaned.dropna().reset_index(drop=True)
        log.append(f"Removed {before - len(cleaned)} rows containing missing values.")

    elif missing_strategy == "Fill numerical with median, text with mode":
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue

            if pd.api.types.is_numeric_dtype(cleaned[column]):
                median = cleaned[column].median()

                if pd.notna(median):
                    cleaned[column] = cleaned[column].fillna(median)
            else:
                mode = cleaned[column].mode(dropna=True)

                if not mode.empty:
                    cleaned[column] = cleaned[column].fillna(mode.iloc[0])

        log.append("Filled missing values where a suitable median or mode existed.")

    elif missing_strategy == "Fill numerical with mean, text with mode":
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue

            if pd.api.types.is_numeric_dtype(cleaned[column]):
                mean = cleaned[column].mean()

                if pd.notna(mean):
                    cleaned[column] = cleaned[column].fillna(mean)
            else:
                mode = cleaned[column].mode(dropna=True)

                if not mode.empty:
                    cleaned[column] = cleaned[column].fillna(mode.iloc[0])

        log.append("Filled missing values where a suitable mean or mode existed.")

    missing_after = int(cleaned.isna().sum().sum())

    log.append(
        f"Missing values: {missing_before} before handling, "
        f"{missing_after} afterward."
    )

    log.append(
        f"Dataset dimensions: {initial_rows} rows × {initial_columns} columns "
        f"before cleaning; {len(cleaned)} rows × {len(cleaned.columns)} columns "
        "after cleaning."
    )

    return cleaned, log
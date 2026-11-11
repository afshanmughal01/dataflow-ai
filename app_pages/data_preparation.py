
import streamlit as st
import pandas as pd


# =========================================================
# HELPER: MAKE COLUMN NAMES UNIQUE
# =========================================================

def make_unique_columns(columns):
    """
    Make duplicate column names unique.
    Does not use private Pandas APIs.
    """

    seen = set()
    unique_columns = []

    for column in columns:

        name = str(column).strip()

        if name not in seen:
            unique_columns.append(name)
            seen.add(name)

        else:
            counter = 1
            new_name = f"{name}_{counter}"

            while new_name in seen:
                counter += 1
                new_name = f"{name}_{counter}"

            unique_columns.append(new_name)
            seen.add(new_name)

    return unique_columns


# =========================================================
# DATA PREPARATION PAGE
# =========================================================

def render_data_preparation(df):

    st.markdown(
        '<p class="eyebrow">DATA MANAGEMENT</p>',
        unsafe_allow_html=True
    )

    st.title("Data Preparation")

    st.markdown(
        "Clean, transform, and prepare your dataset "
        "for accurate analysis."
    )

    # -----------------------------------------------------
    # VALIDATE DATASET
    # -----------------------------------------------------

    if not isinstance(df, pd.DataFrame) or df.empty:

        st.warning("Please upload a valid dataset first.")
        return df, []

    # Always start with a valid DataFrame
    cleaned_df = df.copy(deep=True)

    # Get existing prepared data safely
    saved_df = st.session_state.get("prepared_df")

    if isinstance(saved_df, pd.DataFrame):
        cleaned_df = saved_df.copy(deep=True)

    cleaning_log = st.session_state.get("cleaning_log", [])

    # -----------------------------------------------------
    # ORIGINAL DATASET OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Columns",
        f"{len(df.columns):,}"
    )

    col3.metric(
        "Missing Values",
        f"{int(df.isna().sum().sum()):,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )

    st.divider()

    # -----------------------------------------------------
    # CLEANING OPTIONS
    # -----------------------------------------------------

    st.markdown("### Cleaning Operations")

    st.caption(
        "Select the operations you want to apply, "
        "then click Apply Cleaning."
    )

    with st.form("data_cleaning_form"):

        st.markdown("#### 1. Remove Duplicates")

        remove_duplicates = st.checkbox(
            "Remove duplicate rows",
            value=True
        )

        st.markdown("#### 2. Handle Missing Values")

        missing_option = st.selectbox(
            "Choose missing-value treatment",
            [
                "Keep missing values",
                "Remove rows with missing values",
                "Fill numerical values with mean",
                "Fill numerical values with median",
                "Fill all missing values with mode"
            ]
        )

        st.markdown("#### 3. Text Cleaning")

        trim_spaces = st.checkbox(
            "Remove leading and trailing spaces from text"
        )

        standardize_text = st.checkbox(
            "Standardize text values (lowercase)"
        )

        st.markdown("#### 4. Column Names")

        clean_column_names = st.checkbox(
            "Standardize column names",
            value=True
        )

        st.markdown("#### 5. Remove Empty Columns")

        remove_empty_columns = st.checkbox(
            "Remove columns containing only missing values"
        )

        apply_cleaning = st.form_submit_button(
            "Apply Cleaning",
            type="primary",
            width="stretch"
        )

    # -----------------------------------------------------
    # APPLY CLEANING
    # -----------------------------------------------------

    if apply_cleaning:

        cleaned_df = df.copy(deep=True)
        cleaning_log = []

        # 1. Remove duplicate rows
        if remove_duplicates:

            before = len(cleaned_df)

            cleaned_df = cleaned_df.drop_duplicates()

            removed = before - len(cleaned_df)

            cleaning_log.append(
                f"Removed {removed:,} duplicate rows."
            )

        # 2. Handle missing values
        if missing_option == "Remove rows with missing values":

            before = len(cleaned_df)

            cleaned_df = cleaned_df.dropna()

            removed = before - len(cleaned_df)

            cleaning_log.append(
                f"Removed {removed:,} rows containing missing values."
            )

        elif missing_option == "Fill numerical values with mean":

            numeric_cols = cleaned_df.select_dtypes(
                include="number"
            ).columns

            for col in numeric_cols:

                if cleaned_df[col].notna().any():

                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mean()
                    )

            cleaning_log.append(
                "Filled missing numerical values using column means."
            )

        elif missing_option == "Fill numerical values with median":

            numeric_cols = cleaned_df.select_dtypes(
                include="number"
            ).columns

            for col in numeric_cols:

                if cleaned_df[col].notna().any():

                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].median()
                    )

            cleaning_log.append(
                "Filled missing numerical values using column medians."
            )

        elif missing_option == "Fill all missing values with mode":

            for col in cleaned_df.columns:

                modes = cleaned_df[col].mode(dropna=True)

                if not modes.empty:

                    cleaned_df[col] = cleaned_df[col].fillna(
                        modes.iloc[0]
                    )

            cleaning_log.append(
                "Filled missing values using each column's mode."
            )

        else:

            cleaning_log.append(
                "Missing values were left unchanged."
            )

        # 3. Trim text spaces
        if trim_spaces:

            text_cols = cleaned_df.select_dtypes(
                include=["object", "string"]
            ).columns

            for col in text_cols:

                cleaned_df[col] = cleaned_df[col].apply(
                    lambda value: value.strip()
                    if isinstance(value, str)
                    else value
                )

            cleaning_log.append(
                "Removed leading and trailing spaces from text values."
            )

        # 4. Standardize text
        if standardize_text:

            text_cols = cleaned_df.select_dtypes(
                include=["object", "string"]
            ).columns

            for col in text_cols:

                cleaned_df[col] = cleaned_df[col].apply(
                    lambda value: value.lower()
                    if isinstance(value, str)
                    else value
                )

            cleaning_log.append(
                "Converted text values to lowercase."
            )

        # 5. Standardize column names
        if clean_column_names:

            cleaned_df.columns = (
                cleaned_df.columns
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .str.replace(r"[^a-z0-9_]", "", regex=True)
            )

            # Safely handle duplicate column names
            cleaned_df.columns = make_unique_columns(
                cleaned_df.columns
            )

            cleaning_log.append(
                "Standardized column names and handled duplicates."
            )

        # 6. Remove completely empty columns
        if remove_empty_columns:

            before = len(cleaned_df.columns)

            cleaned_df = cleaned_df.dropna(
                axis=1,
                how="all"
            )

            removed = before - len(cleaned_df.columns)

            cleaning_log.append(
                f"Removed {removed:,} completely empty columns."
            )

        # Validate cleaning result
        if not isinstance(cleaned_df, pd.DataFrame):

            st.error("Cleaning failed. Please check your dataset.")
            return df.copy(deep=True), []

        # Save results
        st.session_state["prepared_df"] = cleaned_df.copy(deep=True)

        st.session_state["cleaning_log"] = cleaning_log

        st.success("Data cleaning completed successfully!")

    # -----------------------------------------------------
    # DISPLAY CLEANING RESULTS
    # -----------------------------------------------------

    # Read saved data only if it is a valid DataFrame
    saved_df = st.session_state.get("prepared_df")

    if isinstance(saved_df, pd.DataFrame):

        cleaned_df = saved_df.copy(deep=True)

        cleaning_log = st.session_state.get(
            "cleaning_log", []
        )

        st.divider()

        st.markdown("### Cleaning Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Rows After Cleaning",
            f"{len(cleaned_df):,}"
        )

        result_col2.metric(
            "Missing Values Remaining",
            f"{int(cleaned_df.isna().sum().sum()):,}"
        )

        result_col3.metric(
            "Duplicate Rows Remaining",
            f"{int(cleaned_df.duplicated().sum()):,}"
        )

        # Cleaning activity
        st.markdown("#### Cleaning Activity")

        if cleaning_log:

            for item in cleaning_log:
                st.write(f"✓ {item}")

        else:

            st.info("No cleaning operations have been recorded.")

        # Dataset preview
        st.markdown("#### Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(100),
            width="stretch",
            hide_index=True
        )

        st.caption(
            f"Showing {min(100, len(cleaned_df)):,} "
            f"of {len(cleaned_df):,} rows."
        )

        # Download cleaned dataset
        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            width="stretch"
        )

        return cleaned_df, cleaning_log

    # No cleaning has been applied yet
    return cleaned_df, cleaning_log

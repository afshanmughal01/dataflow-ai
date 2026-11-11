
import pandas as pd
import numpy as np


def generate_business_insights(df):

    # Validate dataset
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "summary": {},
            "key_findings": [],
            "top_performers": [],
            "anomalies": [],
            "recommendations": [],
            "errors": ["Dataset is empty or invalid."]
        }

    data = df.copy()

    findings = []
    top_performers = []
    anomalies = []
    recommendations = []
    errors = []

    # 1. Dataset Overview
    total_rows = len(data)
    total_columns = len(data.columns)

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    summary = {
        "total_records": total_rows,
        "total_columns": total_columns,
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum())
    }

    # 2. Numeric Analysis
    for column in numeric_columns:

        values = pd.to_numeric(
            data[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            continue

        average = float(values.mean())
        maximum = float(values.max())
        minimum = float(values.min())

        findings.append({
            "type": "Numeric Analysis",
            "column": column,
            "finding": (
                f"{column} has an average of {average:,.2f}, "
                f"a minimum of {minimum:,.2f}, "
                f"and a maximum of {maximum:,.2f}."
            )
        })

        # Outlier Detection using IQR
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            lower_limit = q1 - (1.5 * iqr)
            upper_limit = q3 + (1.5 * iqr)

            outliers = values[
                (values < lower_limit) |
                (values > upper_limit)
            ]

            if not outliers.empty:
                anomalies.append({
                    "column": column,
                    "count": int(len(outliers)),
                    "finding": (
                        f"{len(outliers)} unusual values detected "
                        f"in {column} using the IQR method."
                    )
                })

        # Missing values
        missing_count = int(data[column].isna().sum())

        if missing_count > 0:
            recommendations.append(
                f"Review {missing_count} missing values in {column}."
            )

    # 3. Categorical Analysis
    for column in categorical_columns:

        values = data[column].dropna().astype(str)

        if values.empty:
            continue

        counts = values.value_counts()

        if counts.empty:
            continue

        top_category = str(counts.index[0])
        top_count = int(counts.iloc[0])
        percentage = (top_count / len(values)) * 100

        top_performers.append({
            "column": column,
            "category": top_category,
            "count": top_count,
            "percentage": round(percentage, 2)
        })

        findings.append({
            "type": "Category Analysis",
            "column": column,
            "finding": (
                f"{top_category} is the most frequent category "
                f"in {column}, representing {percentage:.2f}% "
                f"of non-missing records."
            )
        })

        if percentage >= 70:
            recommendations.append(
                f"{column} is heavily concentrated in "
                f"{top_category}. Review the category distribution."
            )

    # 4. Correlation Analysis
    if len(numeric_columns) >= 2:

        numeric_data = (
            data[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
        )

        correlation = numeric_data.corr()

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):

                col1 = numeric_columns[i]
                col2 = numeric_columns[j]

                value = correlation.loc[col1, col2]

                if pd.notna(value) and abs(value) >= 0.7:

                    direction = (
                        "positive" if value > 0 else "negative"
                    )

                    findings.append({
                        "type": "Correlation",
                        "column": f"{col1} / {col2}",
                        "finding": (
                            f"{col1} and {col2} have a strong "
                            f"{direction} correlation of {value:.2f}."
                        )
                    })

    # 5. Data Quality Findings
    missing_values = summary["missing_values"]
    duplicate_rows = summary["duplicate_rows"]

    if missing_values > 0:

        findings.append({
            "type": "Data Quality",
            "column": "Dataset",
            "finding": (
                f"The dataset contains {missing_values} missing cells."
            )
        })

        recommendations.append(
            "Review and clean missing values before critical analysis."
        )

    if duplicate_rows > 0:

        findings.append({
            "type": "Data Quality",
            "column": "Dataset",
            "finding": (
                f"The dataset contains {duplicate_rows} duplicate rows."
            )
        })

        recommendations.append(
            "Review duplicate records to avoid incorrect analysis."
        )

    # 6. General Recommendations
    if not numeric_columns:
        recommendations.append(
            "Add numeric metrics to enable deeper statistical analysis."
        )

    if not categorical_columns:
        recommendations.append(
            "Add categorical fields to enable category comparisons."
        )

    if not recommendations:
        recommendations.append(
            "No major issues detected by the current automated checks."
        )

    # Remove duplicate recommendations
    recommendations = list(dict.fromkeys(recommendations))

    return {
        "summary": summary,
        "key_findings": findings,
        "top_performers": top_performers,
        "anomalies": anomalies,
        "recommendations": recommendations,
        "errors": errors
    }

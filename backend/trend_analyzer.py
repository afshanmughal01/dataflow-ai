
import pandas as pd
import numpy as np


def detect_date_columns(df):
    """Identify likely date columns without changing the dataset."""
    date_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        values = series.dropna().astype(str).str.strip()

        if values.empty:
            continue

        # Avoid interpreting ordinary numbers as dates.
        date_like = values.str.contains(
            r"[-/]|[A-Za-z]{3,}",
            regex=True,
            na=False,
        )

        if date_like.mean() < 0.8:
            continue

        sample = values.head(100)

        try:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed",
            )

            if parsed.notna().mean() >= 0.8:
                date_columns.append(column)

        except (ValueError, TypeError, OverflowError):
            continue

    return date_columns


def analyze_trends(
    df,
    date_column,
    value_column,
    frequency="Monthly",
    aggregation="Sum",
):
    """Calculate time-based trends and period comparisons."""

    data = df[[date_column, value_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
        format="mixed",
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )

    # Rows must have a valid date and numeric value.
    data = data.dropna(subset=[date_column, value_column])

    if data.empty:
        raise ValueError(
            "No valid date and numeric value combinations found."
        )

    frequency_map = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "MS",
        "Yearly": "YS",
    }

    rule = frequency_map[frequency]

    aggregation_map = {
        "Sum": "sum",
        "Average": "mean",
        "Count": "count",
        "Minimum": "min",
        "Maximum": "max",
    }

    result = (
        data.set_index(date_column)
        .sort_index()
        .resample(rule)[value_column]
        .agg(aggregation_map[aggregation])
        .reset_index()
    )

    result.columns = ["Date", "Value"]

    # Do not treat empty periods as zero-valued observations.
    period_counts = (
        data.set_index(date_column)
        .resample(rule)[value_column]
        .count()
        .to_numpy()
    )

    result = result.loc[period_counts > 0].copy()

    result["Previous Value"] = result["Value"].shift(1)

    previous = result["Previous Value"]

    result["Growth %"] = np.where(
        previous.notna() & (previous != 0),
        ((result["Value"] - previous) / previous.abs()) * 100,
        np.nan,
    )

    result["Change"] = result["Value"].diff()

    result["Moving Average"] = (
        result["Value"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    result = result.reset_index(drop=True)

    latest = result.iloc[-1]

    growth = latest["Growth %"]

    summary = {
        "Total Records": len(data),
        "Periods": len(result),
        "Total Value": float(data[value_column].sum()),
        "Average Value": float(data[value_column].mean()),
        "Latest Period Value": float(latest["Value"]),
        "Latest Growth %": (
            float(growth) if pd.notna(growth) else None
        ),
        "Highest Period": str(
            result.loc[result["Value"].idxmax(), "Date"].date()
        ),
        "Lowest Period": str(
            result.loc[result["Value"].idxmin(), "Date"].date()
        ),
    }

    return result, summary


def generate_trend_insights(result, summary):
    """Generate factual, rule-based insights."""

    insights = []

    insights.append(
        f"Analysis covers {summary['Total Records']:,} "
        f"valid records across {summary['Periods']} periods."
    )

    insights.append(
        f"The highest period value occurred on "
        f"{summary['Highest Period']}."
    )

    insights.append(
        f"The lowest period value occurred on "
        f"{summary['Lowest Period']}."
    )

    growth = summary["Latest Growth %"]

    if growth is None:
        insights.append(
            "Growth is unavailable because there is no "
            "previous period or its value is zero."
        )
    elif growth > 0:
        insights.append(
            f"The latest period increased by {growth:.2f}% "
            f"compared with the previous observed period."
        )
    elif growth < 0:
        insights.append(
            f"The latest period decreased by {abs(growth):.2f}% "
            f"compared with the previous observed period."
        )
    else:
        insights.append(
            "The latest period is unchanged compared "
            "with the previous observed period."
        )

    return insights

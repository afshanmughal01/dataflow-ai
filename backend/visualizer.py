import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set_theme(style="whitegrid", palette="deep")


def create_histogram(df, column):
    fig, ax = plt.subplots(figsize=(9, 5))

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        ax.text(0.5, 0.5, "No numerical data available",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.histplot(values, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def create_bar_chart(df, column, top_n=10):
    fig, ax = plt.subplots(figsize=(9, 5))

    counts = df[column].fillna("Missing").astype(str).value_counts().head(top_n)

    if counts.empty:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
        ax.set_title(f"Top {top_n} values in {column}")
        ax.set_xlabel("Count")
        ax.set_ylabel(column)

    fig.tight_layout()
    return fig


def create_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(9, 4))

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        ax.text(0.5, 0.5, "No numerical data available",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.boxplot(x=values, ax=ax)
        ax.set_title(f"Outlier overview: {column}")

    fig.tight_layout()
    return fig


def create_correlation_heatmap(df):
    numerical = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10, 7))

    if numerical.shape[1] < 2:
        ax.text(0.5, 0.5, "At least two numerical columns are required",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.heatmap(
            numerical.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Correlation Heatmap")

    fig.tight_layout()
    return fig


def create_missing_chart(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))

    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.barplot(x=missing.values, y=missing.index, ax=ax)
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Missing count")
        ax.set_ylabel("Column")

    fig.tight_layout()
    return fig
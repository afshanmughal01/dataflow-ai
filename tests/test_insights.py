
import sys
from pathlib import Path

# Add the main project directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backend.insights_engine import generate_business_insights


data = {
    "Product": [
        "Laptop", "Mobile", "Tablet", "Laptop",
        "Mobile", "Laptop", "Tablet", "Mobile"
    ],
    "Region": [
        "Lahore", "Karachi", "Lahore", "Islamabad",
        "Karachi", "Lahore", "Islamabad", "Karachi"
    ],
    "Sales": [
        50000, 30000, 20000, 55000,
        35000, 200000, 22000, 40000
    ],
    "Quantity": [5, 3, 2, 6, 4, 20, 2, 5]
}

df = pd.DataFrame(data)

results = generate_business_insights(df)

print("\n========== DATASET SUMMARY ==========")
print(results["summary"])

print("\n========== KEY FINDINGS ==========")
for item in results["key_findings"]:
    print("-", item["finding"])

print("\n========== TOP CATEGORIES ==========")
for item in results["top_performers"]:
    print(item)

print("\n========== ANOMALIES ==========")
for item in results["anomalies"]:
    print("-", item["finding"])

print("\n========== RECOMMENDATIONS ==========")
for item in results["recommendations"]:
    print("-", item)

print("\n========== ERRORS ==========")
print(results["errors"])

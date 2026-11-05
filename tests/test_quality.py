import sys
from pathlib import Path

# Add project root to Python import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from backend.data_quality import analyze_data_quality
# Sample dataset
data = {
    "Name": ["Ali", "Sara ", "Ahmed", None, "Ali"],
    "Age": [25, 30, None, 28, 25],
    "Salary": ["50000", "60000", "70000", "55000", "50000"],
    "City": ["Lahore", "Karachi", "Lahore", "Islamabad", "Lahore"]
}

df = pd.DataFrame(data)

result = analyze_data_quality(df)

print("\nOVERALL QUALITY SCORE")
print(result["overall_score"])

print("\nQUALITY GRADE")
print(result["grade"])

print("\nDATASET SUMMARY")
print(result["summary"])

print("\nCOLUMN QUALITY REPORT")
print(result["column_report"].to_string(index=False))

print("\nDETECTED ISSUES")
print(pd.DataFrame(result["issues"]).to_string(index=False))
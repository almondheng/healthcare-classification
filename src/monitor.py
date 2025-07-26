import pandas as pd
from evidently import Report
from evidently.metrics import DataDriftPreset

# Load reference and current data
ref = pd.read_csv("data/healthcare_dataset.csv")
ref["target"] = (ref["Medical Condition"].str.lower() == "cancer").astype(int)
ref.drop(columns=["Medical Condition"], inplace=True)
curr = pd.read_csv("data/healthcare_production.csv")

# Generate a data drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=curr)
report.save_html("evidently_report.html")
print("Evidently AI monitoring report generated: evidently_report.html")

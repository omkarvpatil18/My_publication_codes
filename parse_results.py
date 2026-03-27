import json
import csv

# Load JSON results
with open("results_100.json", "r") as f:
    data = json.load(f)

# Open CSV file to write
with open("results_100.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Type", "Supplier/Warehouse", "Scenario"])  # header

    # Extract each solution ("witness")
    for call in data.get("Call", []):
        for witness in call.get("Witnesses", []):
            for value in witness.get("Value", []):
                if value.startswith("select("):
                    # primary supplier
                    s = value[len("select("):-1]
                    writer.writerow(["Primary", s, ""])
                elif value.startswith("select_backup("):
                    # backup supplier
                    b = value[len("select_backup("):-1]
                    writer.writerow(["Backup", b, ""])
                elif value.startswith("objective_value("):
                    obj = value[len("objective_value("):-1]
                    writer.writerow(["Objective", obj, ""])

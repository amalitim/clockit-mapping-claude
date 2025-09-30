import pandas as pd

# The specific file mentioned
input_file = "uploads/1759168943_Brock_Latest_ClockIt_Blank.xlsx"
output_file = "uploads/predicted_claude_1759168943_Brock_Latest_ClockIt_Blank.xlsx"

print("=" * 100)
print("SEARCHING FOR TASKS CONTAINING 'Monday' AND 'Holiday'")
print("=" * 100)

# Load files
df_input = pd.read_excel(input_file, engine='openpyxl')
df_output = pd.read_excel(output_file, engine='openpyxl')

# Search for tasks containing both "Monday" and "Holiday"
input_matches = df_input[
    df_input['Task Name'].astype(str).str.contains('Monday', case=False, na=False) &
    df_input['Task Name'].astype(str).str.contains('Holiday', case=False, na=False)
]

print(f"\nMatches in INPUT file: {len(input_matches)}")

if len(input_matches) > 0:
    print("\nTasks containing both 'Monday' AND 'Holiday' in INPUT:")
    for idx, row in input_matches.iterrows():
        print(f"\n  Row {idx + 2} (Excel row):")
        print(f"    Task Name: {row['Task Name']}")
        print(f"    Employees: {row['Employees']}")
        print(f"    Category: {row['Category']}")
        print(f"    Project: {row['Project']}")

# Search in output
output_matches = df_output[
    df_output['Task Name'].astype(str).str.contains('Monday', case=False, na=False) &
    df_output['Task Name'].astype(str).str.contains('Holiday', case=False, na=False)
]

print(f"\n" + "=" * 100)
print(f"Matches in OUTPUT file: {len(output_matches)}")

if len(output_matches) > 0:
    print("\nTasks containing both 'Monday' AND 'Holiday' in OUTPUT:")
    for idx, row in output_matches.iterrows():
        print(f"\n  Row {idx + 2} (Excel row):")
        print(f"    Task Name: {row['Task Name']}")
        print(f"    Employees: {row['Employees']}")
        print(f"    Predicted_Type: {row['Predicted_Type']}")
        print(f"    Confidence: {row['Confidence']:.4f}")

# Also search separately
print(f"\n" + "=" * 100)
print("ADDITIONAL SEARCHES")
print("=" * 100)

# Tasks with "working" in them
working_tasks = df_input[df_input['Task Name'].astype(str).str.contains('working', case=False, na=False)]
print(f"\nTasks containing 'working' in INPUT: {len(working_tasks)}")
if len(working_tasks) > 0 and len(working_tasks) <= 10:
    for idx, row in working_tasks.iterrows():
        print(f"  Row {idx + 2}: '{row['Task Name']}'")

# Tasks with "Half" in them
half_tasks = df_input[df_input['Task Name'].astype(str).str.contains('Half', case=True, na=False)]
print(f"\nTasks containing 'Half' (case-sensitive) in INPUT: {len(half_tasks)}")
if len(half_tasks) > 0:
    for idx, row in half_tasks.iterrows():
        print(f"  Row {idx + 2}: '{row['Task Name']}'")

print("\n" + "=" * 100)
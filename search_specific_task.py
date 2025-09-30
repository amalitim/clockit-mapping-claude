import pandas as pd

# The specific file mentioned
input_file = "uploads/1759168943_Brock_Latest_ClockIt_Blank.xlsx"
output_file = "uploads/predicted_claude_1759168943_Brock_Latest_ClockIt_Blank.xlsx"

search_term = "Half Day - From working on Monday(Holiday)"

print("=" * 100)
print(f"SEARCHING FOR TASK: '{search_term}'")
print("=" * 100)

# Load files
df_input = pd.read_excel(input_file, engine='openpyxl')
df_output = pd.read_excel(output_file, engine='openpyxl')

print(f"\nINPUT FILE: {input_file}")
print(f"Total rows: {len(df_input)}")

# Search in input file
input_matches = df_input[df_input['Task Name'] == search_term]
print(f"\nExact matches found in INPUT: {len(input_matches)}")

if len(input_matches) > 0:
    print("\nMatching rows in INPUT file:")
    for idx, row in input_matches.iterrows():
        print(f"\n  Row {idx + 2} (Excel row):")  # +2 because 0-indexed and header
        print(f"    Employees: {row['Employees']}")
        print(f"    Task Name: {row['Task Name']}")
        print(f"    Category: {row['Category']}")
        print(f"    Project: {row['Project']}")
        print(f"    Billability Status: {row['Billability Status']}")
else:
    print("\n[NOT FOUND] Task not found in input file")

print(f"\n" + "=" * 100)
print(f"OUTPUT FILE: {output_file}")
print(f"Total rows: {len(df_output)}")

# Search in output file
output_matches = df_output[df_output['Task Name'] == search_term]
print(f"\nExact matches found in OUTPUT: {len(output_matches)}")

if len(output_matches) > 0:
    print("\nMatching rows in OUTPUT file:")
    for idx, row in output_matches.iterrows():
        print(f"\n  Row {idx + 2} (Excel row):")  # +2 because 0-indexed and header
        print(f"    Employees: {row['Employees']}")
        print(f"    Task Name: {row['Task Name']}")
        print(f"    Category: {row['Category']}")
        print(f"    Project: {row['Project']}")
        print(f"    Billability Status: {row['Billability Status']}")
        if 'Predicted_Type' in row.index:
            print(f"    Predicted_Type: {row['Predicted_Type']}")
        if 'Confidence' in row.index:
            print(f"    Confidence: {row['Confidence']:.4f}")
else:
    print("\n[NOT FOUND] Task not found in output file")

# Also search for partial matches
print(f"\n" + "=" * 100)
print("SEARCHING FOR PARTIAL MATCHES (contains 'Half Day')")
print("=" * 100)

# Partial search in input
input_partial = df_input[df_input['Task Name'].astype(str).str.contains('Half Day', case=False, na=False)]
print(f"\nPartial matches in INPUT file: {len(input_partial)}")
if len(input_partial) > 0:
    print("\nTasks containing 'Half Day' in INPUT:")
    for idx, row in input_partial.iterrows():
        print(f"  Row {idx + 2}: '{row['Task Name']}'")

# Partial search in output
output_partial = df_output[df_output['Task Name'].astype(str).str.contains('Half Day', case=False, na=False)]
print(f"\nPartial matches in OUTPUT file: {len(output_partial)}")
if len(output_partial) > 0:
    print("\nTasks containing 'Half Day' in OUTPUT:")
    for idx, row in output_partial.iterrows():
        predicted = row.get('Predicted_Type', 'N/A')
        print(f"  Row {idx + 2}: '{row['Task Name']}' -> {predicted}")

print("\n" + "=" * 100)
print("ANALYSIS")
print("=" * 100)

if len(input_matches) > 0 and len(output_matches) == 0:
    print("\n[ERROR] Task exists in INPUT but NOT in OUTPUT - row was dropped!")
elif len(input_matches) > 0 and len(output_matches) > 0:
    print("\n[SUCCESS] Task exists in both INPUT and OUTPUT files")
elif len(input_matches) == 0:
    print("\n[INFO] Task does not exist in INPUT file (might be mistyped?)")

print("=" * 100)
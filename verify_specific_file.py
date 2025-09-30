import pandas as pd
import os

# The specific file mentioned
input_file = "uploads/1759168943_Brock_Latest_ClockIt_Blank.xlsx"
output_file = "uploads/predicted_claude_1759168943_Brock_Latest_ClockIt_Blank.xlsx"

print("=" * 100)
print("VERIFYING SPECIFIC FILE: 1759168943_Brock_Latest_ClockIt_Blank.xlsx")
print("=" * 100)

# Load files
df_input = pd.read_excel(input_file, engine='openpyxl')
df_output = pd.read_excel(output_file, engine='openpyxl')

print(f"\nINPUT FILE:")
print(f"  Path: {input_file}")
print(f"  Exists: {os.path.exists(input_file)}")
print(f"  Size: {os.path.getsize(input_file):,} bytes")
print(f"  Rows: {len(df_input):,}")
print(f"  Columns: {len(df_input.columns)}")

print(f"\nOUTPUT FILE:")
print(f"  Path: {output_file}")
print(f"  Exists: {os.path.exists(output_file)}")
print(f"  Size: {os.path.getsize(output_file):,} bytes")
print(f"  Rows: {len(df_output):,}")
print(f"  Columns: {len(df_output.columns)}")

print(f"\nROW COUNT COMPARISON:")
print(f"  Input rows: {len(df_input):,}")
print(f"  Output rows: {len(df_output):,}")
print(f"  Difference: {len(df_input) - len(df_output):,}")

if len(df_input) == len(df_output):
    print("\n[SUCCESS] Row counts match perfectly!")
else:
    print(f"\n[ERROR] Row counts do not match! Missing {len(df_input) - len(df_output)} rows")

print(f"\nCOLUMN COMPARISON:")
print(f"  Input columns: {list(df_input.columns)}")
print(f"  Output columns: {list(df_output.columns)}")

# Check for Predicted_Type column
if 'Predicted_Type' in df_output.columns:
    print(f"\n[SUCCESS] 'Predicted_Type' column found in output")
    print(f"  Unique predicted types: {df_output['Predicted_Type'].nunique()}")
    print(f"  Predicted type distribution:")
    print(df_output['Predicted_Type'].value_counts())
else:
    print(f"\n[ERROR] 'Predicted_Type' column NOT found in output")

# Check for Confidence column
if 'Confidence' in df_output.columns:
    print(f"\n[SUCCESS] 'Confidence' column found in output")
    print(f"  Avg confidence: {df_output['Confidence'].mean():.4f}")
    print(f"  Min confidence: {df_output['Confidence'].min():.4f}")
    print(f"  Max confidence: {df_output['Confidence'].max():.4f}")
else:
    print(f"\n[ERROR] 'Confidence' column NOT found in output")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)

if len(df_input) == len(df_output) and 'Predicted_Type' in df_output.columns:
    print("\nThe prediction completed successfully with no missing rows.")
    print("All input rows have corresponding predictions in the output file.")
else:
    print("\nThere may be an issue with the prediction process.")

print("=" * 100)
import pandas as pd
import numpy as np

# Load both files
input_file = "uploads/1759168943_Brock_Latest_ClockIt_Blank.xlsx"
output_file = "uploads/predicted_claude_1759168943_Brock_Latest_ClockIt_Blank.xlsx"

print("=" * 80)
print("DEBUGGING ROW MISMATCH ISSUE")
print("=" * 80)

# Load input file
print(f"\nLoading input file: {input_file}")
df_input = pd.read_excel(input_file, engine='openpyxl')
print(f"Input rows: {len(df_input)}")
print(f"Input columns: {list(df_input.columns)}")

# Load output file
print(f"\nLoading output file: {output_file}")
df_output = pd.read_excel(output_file, engine='openpyxl')
print(f"Output rows: {len(df_output)}")
print(f"Output columns: {list(df_output.columns)}")

# Calculate missing rows
missing_rows = len(df_input) - len(df_output)
print(f"\n{'=' * 80}")
print(f"MISSING ROWS: {missing_rows} ({missing_rows / len(df_input) * 100:.2f}% of input)")
print(f"{'=' * 80}")

# Check for NaN values in key columns
print("\n" + "=" * 80)
print("CHECKING FOR NaN/NULL VALUES IN INPUT FILE")
print("=" * 80)

key_columns = ['Employees', 'Task Name', 'Category', 'Project', 'Billability Status']
for col in key_columns:
    if col in df_input.columns:
        nan_count = df_input[col].isna().sum()
        print(f"{col}: {nan_count} NaN values ({nan_count / len(df_input) * 100:.2f}%)")
    else:
        print(f"{col}: COLUMN NOT FOUND")

# Check if all text columns are empty for some rows
print("\n" + "=" * 80)
print("CHECKING FOR COMPLETELY EMPTY TEXT ROWS")
print("=" * 80)

text_cols = [col for col in key_columns if col in df_input.columns]
df_input['all_text_empty'] = True

for col in text_cols:
    # Check if column value is NaN or empty string
    df_input['all_text_empty'] = df_input['all_text_empty'] & (
        df_input[col].isna() | (df_input[col].astype(str).str.strip() == '')
    )

empty_text_rows = df_input['all_text_empty'].sum()
print(f"Rows with ALL text columns empty: {empty_text_rows}")

if empty_text_rows > 0:
    print("\nSample of empty text rows (first 10):")
    empty_df = df_input[df_input['all_text_empty']].head(10)
    print(empty_df[text_cols])

# Check for rows that might have been filtered
print("\n" + "=" * 80)
print("ANALYZING POTENTIAL ROW FILTERING")
print("=" * 80)

# Look at first few rows from both files
print("\nFirst 5 rows of INPUT file (Task Name column):")
print(df_input['Task Name'].head(5))

print("\nFirst 5 rows of OUTPUT file (Task Name column):")
print(df_output['Task Name'].head(5))

# Look at last few rows
print("\nLast 5 rows of INPUT file (Task Name column):")
print(df_input['Task Name'].tail(5))

print("\nLast 5 rows of OUTPUT file (Task Name column):")
print(df_output['Task Name'].tail(5))

# Check if there's a pattern in missing rows
print("\n" + "=" * 80)
print("IDENTIFYING WHICH ROWS ARE MISSING")
print("=" * 80)

# Create a combined text identifier for each row to match
def create_row_id(row):
    """Create unique identifier from available columns"""
    parts = []
    for col in ['Employees', 'Task Name', 'Category', 'Project', 'Billability Status']:
        if col in row.index:
            val = str(row[col]) if pd.notna(row[col]) else 'NA'
            parts.append(val)
    return '|'.join(parts)

df_input['row_id'] = df_input.apply(create_row_id, axis=1)
df_output['row_id'] = df_output.apply(create_row_id, axis=1)

# Find rows that are in input but not in output
missing_mask = ~df_input['row_id'].isin(df_output['row_id'])
missing_rows_df = df_input[missing_mask]

print(f"\nTotal missing rows: {len(missing_rows_df)}")

if len(missing_rows_df) > 0:
    print("\nFirst 20 missing rows:")
    display_cols = [col for col in ['Employees', 'Task Name', 'Category', 'Project'] if col in missing_rows_df.columns]
    print(missing_rows_df[display_cols].head(20))

    # Check for common patterns
    print("\n" + "=" * 80)
    print("ANALYZING PATTERNS IN MISSING ROWS")
    print("=" * 80)

    for col in display_cols:
        nan_in_missing = missing_rows_df[col].isna().sum()
        print(f"{col} - NaN in missing rows: {nan_in_missing} / {len(missing_rows_df)} ({nan_in_missing / len(missing_rows_df) * 100:.1f}%)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Input rows: {len(df_input)}")
print(f"Output rows: {len(df_output)}")
print(f"Missing: {len(df_input) - len(df_output)} rows")
print("=" * 80)
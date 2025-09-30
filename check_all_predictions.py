import pandas as pd
import os
from glob import glob

uploads_folder = "uploads"

# Find all predicted files
predicted_files = glob(os.path.join(uploads_folder, "predicted_claude_*.xlsx"))

print("=" * 100)
print("CHECKING ALL PREDICTED FILES FOR ROW COUNT MISMATCHES")
print("=" * 100)

mismatches = []

for predicted_file in predicted_files:
    # Extract the original filename
    basename = os.path.basename(predicted_file)
    # Remove 'predicted_claude_' prefix
    original_filename = basename.replace('predicted_claude_', '')
    original_file = os.path.join(uploads_folder, original_filename)

    # Check if original file exists
    if not os.path.exists(original_file):
        print(f"\n[WARNING] Original file not found for {basename}")
        continue

    try:
        # Load both files
        df_original = pd.read_excel(original_file, engine='openpyxl')
        df_predicted = pd.read_excel(predicted_file, engine='openpyxl')

        original_rows = len(df_original)
        predicted_rows = len(df_predicted)
        diff = original_rows - predicted_rows

        # Check for mismatch
        if diff != 0:
            print(f"\n[MISMATCH] FOUND:")
            print(f"   File: {original_filename}")
            print(f"   Original rows: {original_rows}")
            print(f"   Predicted rows: {predicted_rows}")
            print(f"   Missing rows: {diff} ({diff / original_rows * 100:.2f}%)")
            mismatches.append({
                'file': original_filename,
                'original_rows': original_rows,
                'predicted_rows': predicted_rows,
                'missing': diff
            })
        else:
            print(f"\n[OK] {original_filename}: {original_rows} rows")

    except Exception as e:
        print(f"\n[ERROR] processing {basename}: {str(e)}")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

if mismatches:
    print(f"\n[ALERT] Found {len(mismatches)} file(s) with row count mismatches:")
    for m in mismatches:
        print(f"   - {m['file']}: {m['missing']} rows missing ({m['missing'] / m['original_rows'] * 100:.1f}%)")
else:
    print("\n[SUCCESS] All predicted files have matching row counts!")

print("=" * 100)
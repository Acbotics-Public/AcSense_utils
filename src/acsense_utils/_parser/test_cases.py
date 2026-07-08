import os
import pandas as pd
import glob

def test_ac_export_row_count():
    reference_file = "/home/acbotics2/AcSense_utils/parsed1/AC17_1_spiadc.csv"
    reference_df = pd.read_csv(reference_file)
    reference_count = len(reference_df)
    print(f"Reference file row count: {reference_count}")

    aco_files = glob.glob("./output/ACO/AC17_1_*.csv")
    if len(aco_files) == 0:
        print("No ACO files found in ./output/ACO/")
        return False

    total_exported = sum(len(pd.read_csv(f)) for f in aco_files)
    print(f"Total rows across {len(aco_files)} ACO chunk files: {total_exported}")

    if total_exported == reference_count:
        print("PASS: Row counts match")
        return True
    else:
        diff = total_exported - reference_count
        print(f"FAIL: Row counts differ by {diff} rows")
        return False
    
if __name__ == "__main__":
    test_ac_export_row_count()
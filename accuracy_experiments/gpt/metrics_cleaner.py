import os
import sys

import pandas as pd

def clean_csv_and_write(old_path, desired_columns, new_file_name="cleaned"):
    # Create new file name
    directory, file_name = os.path.split(old_path)
    base_name, extension = os.path.splitext(file_name)
    new_file_name = new_file_name + "_" + base_name + extension
    new_file_path = os.path.join(directory, new_file_name)

    desired_columns = ["step"] + desired_columns
    
    # Do groupby average based on column step
    df = pd.read_csv(old_path)
    # original_col_order = df.columns
    result = df.groupby(['step'], as_index=False).mean()
    result = result[desired_columns].dropna()

    # Save to new file
    print("Saved cleaned metrics in ", new_file_path)
    result.to_csv(new_file_path, index=False)
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python metrics_cleaner.py <path_to_csv>")
        sys.exit(1)

    input_path = sys.argv[1]

    clean_csv_and_write(input_path, ["val/loss", "val/ppl"], "val")
    clean_csv_and_write(input_path, ["train/loss_step", "train/ppl_step"], "train")
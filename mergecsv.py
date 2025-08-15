import pandas as pd
import glob
import os
from dotenv import load_dotenv

load_dotenv()
# Path to your CSV folder
folder_path = os.getenv("CSV_FOLDER")
print(f"Using folder path: {folder_path}")

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path,"*.csv"))

# See what files were found
print(f"Found {len(csv_files)} CSV files")
print(csv_files)

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Merge all CSV files into a single DataFrame
if dfs:
    all_races = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} racecards into {len(all_races)} total rows")
    all_races.to_csv(os.path.join(folder_path, "merged_results.csv"), index=False, encoding='utf-8-sig')
    print("Merged raca_result saved to merged_race_result.csv")
else:
    print("No CSVs were loaded. Check your folder path and file format.")



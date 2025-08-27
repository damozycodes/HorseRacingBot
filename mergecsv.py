import pandas as pd
import glob
import os
from dotenv import load_dotenv

load_dotenv()

# Path to your CSV folder from .env
folder_path = os.getenv("CSV_FOLDER")
print(f"Using folder path: {folder_path}")

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)  
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

def add_past_performance(df):
    # Ensure date is datetime for sorting
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Sort by date within each horse's history
    df = df.sort_values('date')

    # Convert positions to numeric (ignore non-numbers)
    df['position'] = pd.to_numeric(df['position'], errors='coerce')

    # Create list of past 5 results before each race
    df['prev_positions'] = (
        df['position']
        .shift(1)  # shift so we don't include current race
        .rolling(5)
        .apply(lambda x: ",".join(map(str, x.dropna().astype(int))), raw=False)
    )

    return df

if dfs:
    # Merge all racecards into one DataFrame
    all_races = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(dfs)} racecards into {len(all_races)} total rows")

    # Clean horse names for consistency
    all_races['horse'] = all_races['horse'].str.strip().str.lower()

    # Apply feature engineering per horse
    all_races = all_races.groupby('horse', group_keys=False, sort=False).apply(add_past_performance)


    # Save merged file
    output_path = os.path.join(folder_path, "merged_results.csv")
    all_races.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Merged race results with features saved to {output_path}")

else:
    print("No CSVs were loaded. Check your folder path and file format.")

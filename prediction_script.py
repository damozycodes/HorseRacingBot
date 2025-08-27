import pandas as pd
from datetime import datetime
import requests
from model import HorseRacingPredictor
import os

DATE = datetime.now().strftime('%Y-%m-%d')

CSV_FOLDER = os.getenv("CSV_FOLDER", r"C:\Users\damozy\Projects\HorseRacingBot\results")
OUTPUT_DIR = CSV_FOLDER
API_URL = "https://api.theracingapi.com/v1/racecards/"

USERNAME = os.getenv("Username") 
PASSWORD = os.getenv("Password")

def fetch_racecard():
    print(f"Fetching racecard for date: {DATE}")
    try:
        headers = {"Accept": "application/json"}
        response = requests.get(API_URL, headers=headers, auth=(USERNAME, PASSWORD), timeout=15)
        print("Status code:", response.status_code)
        print("Response content:", response.text[:500])  # Print first 500 characters of the response
        racecard_data = response.json()

        filtered_racecards = [
            racecard for racecard in racecard_data.get("racecards", [])
            if racecard.get("date") == DATE
        ]

        all_runners = []
        for race in filtered_racecards:
            for runner in race.get("runners", []):
                runner_data = {**race, **runner}
                runner_data["race_id"] = race.get("race_id")
                runner_data.pop("runners", None)
                all_runners.append(runner_data)

        if not all_runners:
            print("No racecards found for today.")
            return None

        df = pd.DataFrame(all_runners)
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"Racecard fetched successfully with {len(df)} horses.")
        preds_path = os.path.join(OUTPUT_DIR, f"racecards_{DATE}.csv")
        df.to_csv(preds_path, index=False)
        
        return df
    except Exception as e:
        print("API request failed:", e)
        return None
    


if __name__ == "__main__":
    predictor = HorseRacingPredictor()

    # Try loading model, if missing -> train & save
    if not predictor.load_model():
        print("Training new model...")
        df_train = pd.read_csv("merged_results.csv")  # your historical dataset
        df_train = predictor.load_and_prepare_data(df_train)
        predictor.train_model(df_train)
        predictor.save_model()

    # Fetch today’s races
    racecards = fetch_racecard()
    if racecards is not None:
        available_races = racecards["race_id"].unique()
        print(f"Found {len(available_races)} races today: {available_races}\n")

        all_predictions = []

        for race_id in available_races:
            print(f" Predictions for Race: {race_id}")
            # Race-level probabilities
            race_probs = predictor.predict_race_probabilities(racecards, race_id)
            if race_probs is not None:
                race_probs["race_id"] = race_id
                all_predictions.append(race_probs)

        # Horse-level probabilities
        race_probs2 = predictor.predict_win_probability(racecards)

        # SAVE predictions automatically
        if all_predictions:
            combined_preds = pd.concat(all_predictions, ignore_index=True)
            preds_path = os.path.join(OUTPUT_DIR, f"race_level_predictions_{DATE}.csv")
            combined_preds.to_csv(preds_path, index=False)
            print(f"Race-level predictions saved to {preds_path}")

        if race_probs2 is not None:
            horse_preds_path = os.path.join(OUTPUT_DIR, f"horse_level_predictions_{DATE}.csv")
            race_probs2.to_csv(horse_preds_path, index=False)
            print(f"Horse-level predictions saved to {horse_preds_path}")
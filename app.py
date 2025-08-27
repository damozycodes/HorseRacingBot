import streamlit as st
import pandas as pd
from datetime import datetime
import os
from model import HorseRacingPredictor
import requests

# --- CONFIG ---
DATE = datetime.now().strftime('%Y-%m-%d')
API_URL = "https://api.theracingapi.com/v1/racecards/"
USERNAME = "amoIybJe6NIYBRRU3DNfIsCW"
PASSWORD = "8vm39wjzAwvKj3Tt6VbpesJb"

# Streamlit UI
st.set_page_config(page_title=f"Horse Race Predictor for", layout="wide")
st.title(f"🏇 Daily Horse Race Predictions {DATE}")

# --- Fetch racecards ---
def fetch_racecard():
    try:
        headers = {"Accept": "application/json"}
        response = requests.get(API_URL, headers=headers, auth=(USERNAME, PASSWORD), timeout=15)
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
            return None

        df = pd.DataFrame(all_runners)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None

# --- Prediction function ---
def run_predictions():
    predictor = HorseRacingPredictor()

    # Load or train model
    if not predictor.load_model():
        df_train = pd.read_csv("merged_results.csv")  # your historical dataset
        df_train = predictor.load_and_prepare_data(df_train)
        predictor.train_model(df_train)
        predictor.save_model()

    racecards = fetch_racecard()
    if racecards is None:
        st.warning("No racecards found for today.")
        return None, None

    available_races = racecards["race_id"].unique()
    all_predictions = []

    for race_id in available_races:
        race_probs = predictor.predict_race_probabilities(racecards, race_id)
        if race_probs is not None:
            race_probs["race_id"] = race_id
            all_predictions.append(race_probs)

    race_level_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else None
    horse_level_df = predictor.predict_win_probability(racecards)

    return race_level_df, horse_level_df

# --- UI Button ---
if st.button("Run Daily Prediction"):
    with st.spinner("Running predictions... Please wait ⏳"):
        race_level, horse_level = run_predictions()

    if race_level is not None:
        st.subheader("📊 Race-level Predictions")
        st.dataframe(race_level)
        st.download_button(
            "⬇️ Download Race Predictions CSV",
            race_level.to_csv(index=False),
            file_name=f"race_level_predictions_{DATE}.csv",
            mime="text/csv"
        )

    if horse_level is not None:
        st.subheader("📊 Horse-level Predictions")
        st.dataframe(horse_level)
        st.download_button(
            "⬇️ Download Horse Predictions CSV",
            horse_level.to_csv(index=False),
            file_name=f"horse_level_predictions_{DATE}.csv",
            mime="text/csv"
        )

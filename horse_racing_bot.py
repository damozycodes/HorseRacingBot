import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import requests
import joblib
from requests.auth import HTTPBasicAuth
from datetime import datetime
import os
import ast
import warnings
from dotenv import load_dotenv
import random

load_dotenv()

# Configuration
DATE = datetime.now().strftime('%Y-%m-%d')
API_URL = "https://api.theracingapi.com/v1/racecards/free"
MODEL_PATH = "horse_racing_model.joblib"
PROB_THRESHOLD = 0.2
MIN_EV = 0.1
TRAINER_JOCKEY_WIN_RATE_THRESHOLD = 0.15

USERNAME = os.getenv("Username")
PASSWORD = os.getenv("Password")
print("Loaded credentials:", USERNAME is not None, PASSWORD is not None)

warnings.filterwarnings("ignore")

# Inside fetch_racecard()
def fetch_racecard():
    print(f"Fetching racecard for date: {DATE}")
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
            }
        response = requests.get(
        f"{API_URL}",
        headers=headers,
        auth=(USERNAME, PASSWORD),
        timeout=10
        )
        print("Status code:", response.status_code)

        racecard_data = response.json()
        filtered_racecards = [
        racecard for racecard in racecard_data.get("racecards", [])
        if racecard.get("date") == DATE
    ]   
        # flattern runners and attach race level data
        all_runners = []
        for race in filtered_racecards:
            for runner in race.get("runners", []):
                runner_data = {
                        **race,
                        **runner
                    }
                del runner_data['runners']  # Remove nested runners
                all_runners.append(runner_data)
        if not all_runners:
            print("No racecards found for today.")
            return None

        racecard= pd.DataFrame(all_runners)
        if 'last_5_positions' not in racecard.columns or racecard['last_5_positions'].isnull().all():
            racecard['last_5_positions'] = [[random.randint(1, 10) for _ in range(5)] for _ in range(len(racecard))]

        if 'jockey_win_rate_30d' not in racecard.columns or racecard['jockey_win_rate_30d'].isnull().all():
            racecard['jockey_win_rate_30d'] = np.round(np.random.uniform(0.05, 0.30, size=len(racecard)), 3)

        if 'trainer_win_rate_30d' not in racecard.columns or racecard['trainer_win_rate_30d'].isnull().all():
            racecard['trainer_win_rate_30d'] = np.round(np.random.uniform(0.05, 0.25, size=len(racecard)), 3)

        if 'horse_win_rate_going' not in racecard.columns or racecard['horse_win_rate_going'].isnull().all():
            racecard['horse_win_rate_going'] = np.round(np.random.uniform(0.05, 0.40, size=len(racecard)), 3)

        if racecard.empty:
            print("No racecards found for today.")
            return None

        racecard = racecard.rename(columns={
            'horse': 'horse_name',
            'off_time': 'race_time', 
            'distance_f': 'distance_furlongs'
        })

        print(f"Racecard fetched successfully with {len(racecard)} horses.")
        # save_selections(racecard)
        return racecard

    except Exception as e:
        print("API credentials are invalid or request failed (using fallback dummy data)")
        return generate_dummy_racecard()


def generate_dummy_racecard():
    print("Using fallback data due to API authentication issues")
    return pd.DataFrame({
        'horse_name': ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E'],
        'race_time': ['14:30'] * 5,
        'course': ['Newmarket'] * 5,
        'distance_furlongs': [7] * 5,
        'going': ['Good'] * 5,
        'jockey': ['Jockey 1', 'Jockey 2', 'Jockey 3', 'Jockey 4', 'Jockey 5'],
        'trainer': ['Trainer 1', 'Trainer 2', 'Trainer 3', 'Trainer 4', 'Trainer 5'],
        'last_5_positions': [[1, 2, 3, 4, 5], [3, 4, 2, 1, 6], [2, 1, 5, 3, 2], [4, 3, 2, 1, 5], [5, 4, 3, 2, 1]],
        'jockey_win_rate_30d': [0.15, 0.10, 0.22, 0.18, 0.12],
        'trainer_win_rate_30d': [0.12, 0.14, 0.09, 0.13, 0.11],
        'horse_win_rate_going': [0.3, 0.15, 0.25, 0.2, 0.18],
        'odds': [3.5, 5.0, 4.2, 6.0, 7.1],
        'race_type': ['Flat'] * 5,
    })


def load_historical_data():
    print("Loading historical data...")
    try:
        df = pd.read_csv("horse_racing_data.csv")
        required_columns = [
            'horse_name', 'race_date', 'course', 'distance_furlongs', 'going',
            'jockey', 'trainer', 'last_5_positions', 'jockey_win_rate_30d',
            'trainer_win_rate_30d', 'horse_win_rate_going', 'odds', 'did_win'
        ]
        df['did_win'] = df['did_win'].astype(int)

        # Safely parse stringified lists into real lists
        def safe_parse_positions(x):
            try:
                if isinstance(x, str):
                    return [int(n) for n in ast.literal_eval(x)]
                elif isinstance(x, list):
                    return [int(n) for n in x]
                else:
                    return []
            except Exception:
                return []

        df['last_5_positions'] = df['last_5_positions'].apply(safe_parse_positions)

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Historical data missing required columns")

        print("Historical data loaded successfully.")
        return df

    except Exception as e:
        print("Historical data not found or invalid. Using dummy training data.")
        return pd.DataFrame({
            'horse_name': ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E'],
            'race_date': ['2025-06-01'] * 5,
            'course': ['Newmarket'] * 5,
            'distance_furlongs': [7, 7, 7, 7, 7],
            'going': ['Good'] * 5,
            'jockey': ['Jockey 1', 'Jockey 2', 'Jockey 3', 'Jockey 1', 'Jockey 2'],
            'trainer': ['Trainer 1', 'Trainer 2', 'Trainer 3', 'Trainer 5', 'Trainer 28'],
            'last_5_positions': [[1, 2, 3, 4, 5], [3, 4, 2, 1, 6], [2, 1, 5, 3, 2], [4, 3, 2, 1, 5], [5, 4, 3, 2, 1]],
            'jockey_win_rate_30d': [0.2, 0.1, 0.25, 0.2, 0.6],
            'trainer_win_rate_30d': [0.18, 0.12, 0.22, 0.15, 0.2],
            'horse_win_rate_going': [0.3, 0.15, 0.25, 0.2, 0.18],
            'odds': [4.0, 5.0, 3.5, 6.0, 7.1],
            'did_win': [1, 0, 0, 1, 0] 
            })

def preprocess_data(data, is_training=True):
    data = data.copy()

    data['avg_last_5_positions'] = data['last_5_positions'].apply(lambda x: np.mean(x) if x else 5.0)

    data['form_score'] = data['last_5_positions'].apply(lambda x: sum(1 / (pos + 1) for pos in x) if x else 0)

    features = [
        'distance_furlongs',
        'avg_last_5_positions',
        'form_score',
        'jockey_win_rate_30d',
        'trainer_win_rate_30d',
        'horse_win_rate_going'
    ]

    X = data[features]
    X = (X - X.mean()) / X.std()

    if is_training:
        y = data['did_win']
        return X, y
    return X

def train_model():
    print("Training model...")
    data = load_historical_data()
    X, y = preprocess_data(data, is_training=True)

    if y.sum() == 0:
        raise ValueError("Training data must have at least one positive sample.")

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=6
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading existing model...")
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print("Error loading model: {e}. Training new model...")
    return train_model()


def predict_selection(model, racecard):
    print("Analyzing horses...")
    features = preprocess_data(racecard, is_training=False)
    probabilities = model.predict_proba(features)[:, 1]
    racecard['win_probability'] = probabilities
    racecard['expected_value'] = (racecard['win_probability'] * racecard['odds']) - 1
    return racecard

def select_horses(racecard, probabilities):
    ground_condition = racecard['going'].iloc[0] if not racecard.empty else 'Good'
    print(f"Ground condition for today: {ground_condition}")

    selections = []
    for i, prob in enumerate(probabilities):
        odds = racecard['odds'].iloc[i]
        ev = (prob * odds) - 1
        ground_match = racecard['going'].iloc[i] == ground_condition
        trainer_form = racecard['trainer_win_rate_30d'].iloc[i] >= TRAINER_JOCKEY_WIN_RATE_THRESHOLD
        jockey_form = racecard['jockey_win_rate_30d'].iloc[i] >= TRAINER_JOCKEY_WIN_RATE_THRESHOLD

        if (prob >= PROB_THRESHOLD and ev >= MIN_EV and ground_match and trainer_form and jockey_form):
            selections.append({
                'horse_name': racecard['horse_name'].iloc[i],
                'race': f"{racecard['race_time'].iloc[i]} {racecard['course'].iloc[i]}",
                'probability': round(prob, 3),
                'odds': odds,
                'ev': round(ev, 3),
                'jockey': racecard['jockey'].iloc[i],
                'trainer': racecard['trainer'].iloc[i]
            })

    return pd.DataFrame(selections)


def save_selections(df):
    filename = f"selections_{DATE}.csv"
    df.to_csv(filename, index=False)
    print(f"Selections saved to {filename}")


def main():
    print(f"Starting horse racing analysis for {DATE}")
    racecard = fetch_racecard()
    
    if racecard is None:
        print("No racecard available today. Exiting.")
        return

    model = get_model()
    analyzed = predict_selection(model, racecard)
    selections = select_horses(analyzed, analyzed['win_probability'])

    if not selections.empty:
        print(selections[['horse_name', 'probability', 'ev']])
    else:
        print("No suitable selections found.")
    
    save_selections(selections)

if __name__ == "__main__":
    main()
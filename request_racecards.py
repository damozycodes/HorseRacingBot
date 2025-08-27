
import os
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import time
import pandas as pd

load_dotenv()

USERNAME = os.getenv("Username") 
PASSWORD = os.getenv("Password")

today = time.strftime("%y-%m-%d")
BASE_URL = "https://api.theracingapi.com/v1/racecards"

def fetch_race_cards():
    """function to fect daily race cards from the racing API"""
    try:
        headers = {
            'Accept': 'application/json',
            "Content_type": 'application/json'
        }

        response = requests.get(
            BASE_URL,
            headers=headers,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            timeout=10
        )
        print("Status code:", response.status_code)
        # print(response.text)
        result = response.json()
        racecards = result.get("racecards", [])
        if not racecards:
            print("No racecards found for today{today}.")
            return None
        
        all_racecards = []
        for racecard in racecards:
            for runner in racecard.get("runners", []):
                runner_data = {
                    **racecard,
                    **runner
                }
                runner_data.pop('runners', None)
                all_racecards.append(runner_data)
        if not all_racecards:  
            print(f"No racecards found for today: {today}. and nothing was proceed")
            return None
        return all_racecards

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_daily_racecards(racecards):
    """function to save daily racecards to a CSV file"""

    if not racecards:
        print("No racecards to save.")
        return None

    try:
        df = pd.DataFrame(racecards)
        filename = f"racecards_{today}.csv"
        df.to_csv(filename, index=False)
        print(f"Racecards saved to {filename}")
        
    except Exception as e:
        print(f"An error occurred while saving racecards: {e}")
        return None
    

racecards = fetch_race_cards()
save_daily_racecards(racecards)
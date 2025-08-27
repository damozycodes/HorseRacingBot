import os
import re
import time
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import requests
import pandas as pd

load_dotenv()

API_URL = "https://api.theracingapi.com/v1/results/today?limit=50"
USERNAME = os.getenv("username")
PASSWORD = os.getenv("password")

today = time.strftime("%Y-%m-%d")

def pull_daily_results():
    """function to pull daily results from the racing API"""

    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
            }

        response = requests.get(
            API_URL,
            headers=headers,
            auth=HTTPBasicAuth(USERNAME, PASSWORD),
            timeout=10
            )
        print ("Status code:", response.status_code)
        print (response.text)
        print(len(response.text))

        raw_result= response.json()
        print("Top-level keys:", raw_result.keys())
        results = raw_result.get("results", [])
        print(type(results))

        if not results:
            print("No results found for today.")
            return None
        all_results = []
        for result in results:
            for runner in result.get("runners", []):
                runner_data = {
                    **result,
                    **runner
                }
                runner_data.pop('runners', None) # remove nested runners
                all_results.append(runner_data)
            if not all_results:
                print("No results found for today:{today}. and nothing was proceed")
                return None

        print("Results fetched and processed successfully.")
        return all_results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'response' in locals():
            try:
                response.raise_for_status()
            except:
                pass
        return None
    
def save_results_to_file(results):
    """function to save results to a file"""

    if not results:
        print("No results to save.")
        return None
    
    try:
        df = pd.DataFrame(results)
        filename = f"racing_results_for_{today}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving results: {e}")
        return None
all_results = pull_daily_results()
save_results_to_file(all_results)
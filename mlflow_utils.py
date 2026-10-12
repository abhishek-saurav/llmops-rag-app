import mlflow 
import json 
from pathlib import Path 


# SET PATH FOR THE JSON FILE 
ROOT_DIR = Path(__file__).parent 
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"


def log_run_info(run_id: str, run_name: str): 
    if JSON_FILE_PATH.exists(): 
        with open(JSON_FILE_PATH, "r") as file: 
            historical_run = json.load(file) 

    else: 
        historical_run = []

    run_dict = {"run_id": run_id, 
                "run_name": run_name}

    historical_run.append(run_dict)

    with open(JSON_FILE_PATH, "w") as file: 
        json.dump(historical_run, file, indent=4) 




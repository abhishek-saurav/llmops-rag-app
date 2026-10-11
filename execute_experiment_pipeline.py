from data.generate_eval_datasetv2 import generate_evaluation_dataset 
from evals.application_evals.evaluate_rag_app import evaluate_app
from config.config import params_config 
from dotenv import load_dotenv 
from pathlib import Path 
from langfuse import get_client
from typing import Literal
import logging
from logging import INFO
import mlflow 
import json 
from time import sleep
# LOAD THE API KEYS 
load_dotenv() 

ROOT_DIR = Path(__file__).parent

def flatten_params(param_dict: dict) -> dict: 

    config_dict = dict() 
    for key, value in param_dict.items(): 
        if isinstance(value, dict): 
            config_dict.update(flatten_params(value)) 

        else: 
            if key in config_dict: 
                raise ValueError(f"Duplicate Key Found {key}")

            config_dict[key] = value 

    return config_dict 

def get_latest_result(path: Path | str): 
    filenames = []
    files = Path(path).glob("*.json")
    for file in files: 
        filenames.append(file.stem)

    # print(filenames)
    return (max(filenames))


# IMPORTING DATA FROM EVALUATION RESULT - A JSON FILE 
def get_metrics_from_results(result_json) -> dict: 
    metrics = {}
    with open (result_json, "r") as file: 
        metric_result = json.load(file)['metricsScores']
    for result in metric_result: 
        metric_name = result['metric'].removesuffix("[GEval]").strip() if "[GEval]" in result['metric'] else result['metric'].strip()
        scores = result['scores']
        avg_scores = round((sum(scores)/ len(scores)), 2)

        metrics[metric_name] = avg_scores

    return metrics


def import_system_prompt(label = "staging") -> str: 
    langfuse = get_client() 

    prompt = langfuse.get_prompt(
        name = "rag_app_system_prompt", 
        type = "text" , 
        label= label
    )

    return prompt.prompt 


def artifact_name(artifact_type: Literal['eval_dataset_final', 'golden_dataset_final'], artifact_dir) -> tuple[str, str]: 
    DATA_PATH = ROOT_DIR / "data" / "evaluation"
    
    if artifact_type == 'eval_dataset_final': 
        artifact_path = (DATA_PATH / "eval_dataset" / params_config.evaluation_dataset.evaluation_dataset_filename).as_posix() + ".json"

        return (artifact_path, artifact_dir)

    elif artifact_type == 'eval_dataset_final': 
        artifact_path = (DATA_PATH / "goldens" / params_config.golden_dataset.golden_dataset_filename).as_posix() + ".json"

        return (artifact_path, artifact_dir)


def return_code_files(): 
    code_file_paths = ROOT_DIR / "src" 

    app_code_path = code_file_paths/ "app"/ "rag_app.py"
    eval_data_path = code_file_paths/ "data" / "generate_eval_datasetv2.py"
    evaluation_path = code_file_paths / "evals" / "application_evals" / "evaluate_rag_app.py"
    golden_dataset_path = code_file_paths / "data" / "generate_goldensv2.py"

    return [file.as_posix() for file in [app_code_path, eval_data_path, evaluation_path, golden_dataset_path]] 



if __name__ == "__main__": 

    # SET TRACKING SERVER 
    mlflow.set_tracking_uri("http://localhost:5000")

    # SET THE EXPERIMENT NAME 
    mlflow.set_experiment("rag_app")

    # DO THE LOGGING 
    logger = logging.getLogger(name = "MLFlow Logger") 

    # ADD STREAM HANDLER 
    handler = logging.StreamHandler() 
    logger.addHandler(handler)
    logger.setLevel(INFO)
    formatter = logging.Formatter(fmt = "%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt = formatter)


    with mlflow.start_run() as run: 

        # GET ALL THE PARAMS 
        all_params = params_config.model_dump()
        params_dict = flatten_params(all_params)

        mlflow.log_params(params_dict)
        logger.info("Params logged")

        # GENERATE EVALUATION DATA 
        generate_evaluation_dataset() 
        logger.info("Evaluation Dataset Created")

        # RUN THE EVAL PIPELINE 
        evaluate_app()
        logger.info("Evaluation Complete")

        # PATHS LATEST REPORTS 
        result_dir = ROOT_DIR / "reports" / "evaluation_result"
        report_dir = ROOT_DIR / "reports" / "evaluation_report"
        

        # GET THE LATEST FILES AFTER EVAL PIPELINE 
        latest_results_filename = get_latest_result(result_dir)
        latest_report_filename = get_latest_result(report_dir)
        report_file = (result_dir / latest_report_filename).with_suffix(".json").as_posix()
        result_file = (result_dir / latest_results_filename).with_suffix(".json").as_posix()

        # LOG THE RESULT 
        mlflow.log_artifact(result_file, "results")
        mlflow.log_artifact(report_file, "reports")
        logger.info("Reports and Results logged")

        # LOG THE METRICS 
        metrics = get_metrics_from_results(result_file)
        mlflow.log_metrics(metrics)

        # LOG THE SYSTEM PROMPT

        system_prompt = import_system_prompt(params_config.rag_app.prompt_label)

        mlflow.log_text(system_prompt, 
                        artifact_file= "system_prompt.txt")


        # LOG THE DATASETS 

        eval_dataset_artifact = artifact_name(artifact_type = "eval_dataset_final", 
                                            artifact_dir = "eval_dataset")

        golden_dataset_artifact = artifact_name(artifact_type= "golden_dataset_final", 
                                                artifact_dir= "golden_dataset")

        mlflow.log_artifact(eval_dataset_artifact[0], eval_dataset_artifact[1])
        mlflow.log_artifact(golden_dataset_artifact[0], golden_dataset_artifact[1])
        logger.info("Logged evaluation and golden datasets ")

        # LOG CODE FILES 
        code_files = return_code_files() 

        for code_file in code_files: 
            mlflow.log_artifact(code_file, "code")

        logger.info("code files logged")
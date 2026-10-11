# from config.config import params_config 
# import pprint 
from pathlib import Path
# import json 
# # pprint.pprint(params_config.model_dump())
# from dotenv import load_dotenv


# load_dotenv() 


# def flatten_params(param_dict: dict) -> dict: 

#     config_dict = dict() 
#     for key, value in param_dict.items(): 
#         if isinstance(value, dict): 
#             config_dict.update(flatten_params(value)) 

#         else: 
#             if key in config_dict: 
#                 raise ValueError(f"Duplicate Key Found {key}")

#             config_dict[key] = value 

#     return config_dict 

# def get_latest_result(path: Path | str): 
#     filenames = []
#     files = Path(path).glob("*.json")
#     for file in files: 
#         filenames.append(file.stem)

#     # print(filenames)
#     return (max(filenames))

# ROOT_DIR = Path(__file__).parent
# evaluation_result_dir = ROOT_DIR / "reports" / "evaluation_result"

# recent_file = get_latest_result(evaluation_result_dir) 


# def get_metrics_from_results(result_json) -> dict: 
#     metrics = {}
#     with open (result_json, "r") as file: 
#         metric_result = json.load(file)['metricsScores']
#     for result in metric_result: 
#         metric_name = result['metric'].removesuffix("[GEval]").strip() if "[GEval]" in result['metric'] else result['metric'].strip()
#         scores = result['scores']
#         avg_scores = round((sum(scores)/ len(scores)), 2)

#         metrics[metric_name] = avg_scores

#     return metrics
    
# print() 
# print()
# print(evaluation_result_dir)

# filename = (evaluation_result_dir / recent_file).as_posix() + ".json"

# metrics = get_metrics_from_results(filename)
# pprint.pprint(metrics)

# from langfuse import get_client 

# def import_system_prompt(label = "staging") -> str: 
#     langfuse = get_client() 

#     prompt = langfuse.get_prompt(
#         name = "rag_app_system_prompt", 
#         type = "text" , 
#         label= label
#     )

#     return prompt.prompt 

# prompt = import_system_prompt(params_config.rag_app.prompt_label)
# print(prompt)

path = Path(__file__).parent / "reports" / "evaluation_report"
print(path)
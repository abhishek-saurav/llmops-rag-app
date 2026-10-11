import yaml 
from pathlib import Path 
from config.config_types import Params
import pprint 


# DEFINE PATHS 
ROOT_DIR = Path(__file__).resolve().parent.parent.parent 
PARAMS_PATH = ROOT_DIR/ "params.yaml"

# LOAD THE CONFIG FILE 
def load_config(params_path: Path | str):
    with open (params_path, "r") as file: 
        yaml_config = yaml.safe_load(file) 

    # VALIDATE THE PARAMS 
    params_config = Params.model_validate(obj = yaml_config, 
                          strict= True)

    return params_config

params_config = load_config(PARAMS_PATH)
# pprint.pprint(params_config.model_dump())
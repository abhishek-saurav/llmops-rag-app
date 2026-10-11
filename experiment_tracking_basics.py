import mlflow 
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# SET THE TRACKING SERVER 
mlflow.set_tracking_uri("http://localhost:5000")

# SET THE EXPERIMENT RUN  
mlflow.set_experiment("rag_app")

# START A RUN 
with mlflow.start_run(run_name = "test_run") as run: 
    # PARAMS 
    mlflow.log_param("chunk_size", 300) 
    mlflow.log_param("chunk_overlap", 30) 
    mlflow.log_param("output_dims", 1024)

    # METRICS 
    mlflow.log_metric("recall", 0.89) 
    mlflow.log_metric("faithfulness", 0.91)
    mlflow.log_metric("answer_correctness", 0.7)  

    # ARTIFACTS 
    mlflow.log_artifact(local_path=(ROOT_DIR/ "src" / "app"/ "rag_app.py").as_posix(), 
                        artifact_path="code")

    # DATASETS 
    mlflow.log_artifact(local_path= (ROOT_DIR / "data"/ "evaluation" / "goldens"/ "golden_dataset_final.json").as_posix())
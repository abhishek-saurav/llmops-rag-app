from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase 
from dotenv import load_dotenv 
from pathlib import Path 
from app.rag_app import graph
from time import sleep

# LOAD THE API KEY 
load_dotenv() 

def generate_evaluation_dataset(): 

    # CREATE THE PATHS 
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    GOLDENS_PATH = ROOT_DIR / "data" / "evaluation" / "goldens" / "golden_dataset_final.json"

    # CREATE DIRECTORY 
    EVALUATION_DATA_DIR = ROOT_DIR / "data" / "evaluation" / "eval_dataset"
    EVALUATION_DATA_DIR.mkdir(exist_ok= True, parents= True)

    # DATASET TO READ GOLDENS FROM 
    golden_dataset = EvaluationDataset() 
    golden_dataset.add_goldens_from_json_file(file_path= GOLDENS_PATH)

    # DATASET TO HOLD PRODUCED TEST CASES 
    eval_dataset = EvaluationDataset() 


    # INVOKE THE APPLICATION 
    for golden in golden_dataset.goldens: 
        final_state = graph.invoke({"query": golden.input})
        sleep(3) 
        test_case = LLMTestCase(
            input= golden.input,
            actual_output= final_state.get("response"), 
            expected_output= golden.expected_output, 
            retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
        )

        eval_dataset.add_test_case(test_case= test_case)

    # SAVE THE DATASET ALONG WITH THE TEST CASES 
    eval_dataset.save_as(file_type="json", 
                    directory= EVALUATION_DATA_DIR, 
                    file_name= "evaluation_dataset_final", 
                    include_test_cases= True)

        

if __name__ == "__main__": 
    generate_evaluation_dataset() 
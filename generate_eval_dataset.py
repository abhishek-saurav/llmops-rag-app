from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase 
from dotenv import load_dotenv 
from pathlib import Path 
from rag_workflow import graph
from time import sleep

# LOAD THE API KEY 
load_dotenv() 


# CREATE THE PATHS 
ROOT_DIR = Path(__file__).parent 
GOLDENS_PATH = ROOT_DIR / "datasets" / "goldens" / "golden_dataset.json"
EVALUATON_DATA_DIR = GOLDENS_PATH.parent.parent / "eval_dataset"

# CREATE DIRECTORY 
EVALUATON_DATA_DIR.mkdir(exist_ok= True, parents= True)

# CREATE AN EVALUATION DATASET 
dataset = EvaluationDataset()

# ADD GOLDENS TO THE DATASET 
dataset.add_goldens_from_json_file(
    file_path= GOLDENS_PATH
)

# INVOKE THE APPLICATION 
for golden in dataset.goldens: 
    final_state = graph.invoke({"query": golden.input})
    sleep(3) 
    test_case = LLMTestCase(
        input= golden.input,
        actual_output= final_state.get("response"), 
        expected_output= golden.expected_output, 
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
    )

    dataset.add_test_case(test_case= test_case)

# SAVE THE DATASET ALONG WITH THE TEST CASES 
dataset.save_as(file_type="json", 
                directory= EVALUATON_DATA_DIR, 
                file_name= "evaluation_dataset", 
                include_test_cases= True)

    


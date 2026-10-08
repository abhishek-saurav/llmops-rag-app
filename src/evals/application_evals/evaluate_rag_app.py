from deepeval.metrics import (
    GEval, 
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualRecallMetric, 
    ContextualPrecisionMetric, 
    ContextualRelevancyMetric
)

from deepeval.evaluate import evaluate 
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams, LLMTestCase
from deepeval.dataset.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from pathlib import Path
from dotenv import load_dotenv 


# LOAD THE API KEYS 
load_dotenv() 

# DEFINING THE MODEL (to save cost for test purpose)
model = 'gpt-5.4-mini'

#  DEFINE THE METRICS 
recall = ContextualRecallMetric(model = model) 
precision = ContextualPrecisionMetric(model = model) 
contextual_relevancy = ContextualRelevancyMetric(model = model) 
answer_relevancy = AnswerRelevancyMetric(model = model) 
faithfulness = FaithfulnessMetric(
    # truths_extraction_limit= 5, 
    # penalize_ambiguous_claims= True, 
    model = model
)


answer_correctness = GEval(
    name = 'answer correctness', 
    evaluation_params= [SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT], 
    criteria= """Evaluate the LLM Response based on the correctness of the answer. 
    Compare the 'expected output' with the 'actual output'. Penalize wrong facts strictly""", 
    rubric= [
        Rubric(score_range= (0, 5), expected_outcome= "Answer has incorrect facts"), 
        Rubric(score_range= (6,9), expected_outcome= "Answer is mostly correct but has minor differences"), 
        Rubric(score_range= (10,10), expected_outcome= f"100% correct")
        ], 
    model = model
)
simple_explanation = GEval(
    name = "simple explanation", 
    evaluation_params= [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT], 
    evaluation_steps= [
        "Read the 'actual output' first and then the 'input'", 
        "Check weather the response is simple and easy to understand or not", 
        "Make sure the repsonse has least number of technical jagons and is student friendly"
    ], 
    rubric= [ 
        Rubric(score_range=(0, 3), expected_outcome= "Too Difficult"),
        Rubric(score_range=(4, 7), expected_outcome= "Slight Difficult"),
        Rubric(score_range=(8, 9), expected_outcome= "Moderately Simple"),
        Rubric(score_range=(10,10), expected_outcome= "Very Simple")
    ], 
    model = model
)

# DEFINE THE DATASET PATH 

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "evaluation" / "eval_dataset" / "evaluation_dataset_final.json"

if DATASET_PATH: 
    # LOAD THE DATASET 
    dataset = EvaluationDataset() 

    # LOAD THE TEST CASES 
    dataset.add_test_cases_from_json_file(
        file_path= DATASET_PATH, 
        input_key_name= "input", 
        actual_output_key_name= "actual_output", 
        expected_output_key_name= "expected_output",
        retrieval_context_key_name= "retrieval_context"
    )

    # STORE THE TEST CASES IN A LIST 
    test_cases = dataset.test_cases 


    # EVALUATE THE DATASET 
    evaluate(
        test_cases= test_cases, 
        metrics= [
            recall, 
            precision, 
            answer_correctness, 
            answer_relevancy, 
            faithfulness, 
            contextual_relevancy, 
            simple_explanation
        ], 
        async_config= AsyncConfig(throttle_value= 3, 
                                  max_concurrent= 5),
        display_config= DisplayConfig(
            results_folder= (ROOT_DIR / "reports" / "evaluation_result").as_posix(), 
            file_type= "md",
            file_output_dir= (ROOT_DIR / "reports" / "evaluation_report").as_posix()
        )
    )
from deepeval.metrics import GEval 
from deepeval.test_case import LLMTestCase 
from deepeval.metrics.g_eval import Rubric 
from deepeval.test_case.llm_test_case import SingleTurnParams
from dotenv import load_dotenv 


# LOAD THE API KEY 
load_dotenv() 


# DEFINE THE CUSTOM CRITERIA 
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
    verbose_mode= True 
)


test_case = LLMTestCase(
    input = "What is Linear Regression?", 
    actual_output= "Linear regression is a supervised machine learning and statistical algorithm used to predict a continuous target variable (dependent variable) by modeling its relationship with one or more input features (independent variables) using a best-fit straight line."
)

print(simple_explanation.measure(test_case= test_case))
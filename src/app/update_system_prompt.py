from langfuse import get_client 
from dotenv import load_dotenv 



# LOAD THE API KEYS 
load_dotenv() 

# SHIFTING THE LABELS IN DEV 
langfuse = get_client() 

langfuse.update_prompt(
    name = "rag_app_system_prompt", 
    version= None , 
    new_labels= ['staging']

)
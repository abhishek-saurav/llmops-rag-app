from langfuse import get_client 
from dotenv import load_dotenv 

# LOAD THE API KEYS 
load_dotenv() 

langfuse = get_client() 

system_prompt = langfuse.get_prompt(
    name = "rag_app_system_prompt", 
    type= 'text', 
    label= 'staging'
    )


print(system_prompt.version)
print() 
print(system_prompt.config) 
print() 
print(system_prompt.labels)
print() 
print(system_prompt.prompt)
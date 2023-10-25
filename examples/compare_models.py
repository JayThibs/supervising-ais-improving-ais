from unsupervised_behavioural_clustering.models import GPT4, LLM
from unsupervised_behavioural_clustering.generation import query_model
from unsupervised_behavioural_clustering.analysis import compare_results
from unsupervised_behavioural_clustering.prompts import (
    PROMPT_TEMPLATES,
)  # Importing PROMPT_TEMPLATES


gpt4 = GPT4()
llm = LLM()
gpt4_results = query_model(gpt4, PROMPT_TEMPLATES)
llm_results = query_model(llm, PROMPT_TEMPLATES)
comparison = compare_results(gpt4_results, llm_results)
print(comparison)

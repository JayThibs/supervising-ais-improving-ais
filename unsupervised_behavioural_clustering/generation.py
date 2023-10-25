import openai
from langchain import LLMChain, PromptTemplate


def query_model_on_statements(statements, llm, prompt):
    inputs = []
    responses = []

    chain = LLMChain(llm=llm, prompt=prompt)

    for statement in statements:
        prompt_formatted = chain.format(statement=statement)
        result = llm.generate(prompt_formatted)

        inputs.append(statement)
        responses.append(result)

    return inputs, responses

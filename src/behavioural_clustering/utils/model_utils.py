import time

from behavioural_clustering.models.model_factory import initialize_model


def query_model_on_statements(
    statements, model_family, model_name, prompt_template, system_message, model=None, max_tokens=None
):
    query_results = {}
    inputs, responses, full_conversations = [], [], []
    model_info = {
        "model_family": model_family,
        "model_name": model_name,
        "system_message": system_message
    }
    query_results["model_info"] = model_info
    num_statements = len(statements)

    print("query_model_on_statements...")

    if model is None:
        model = initialize_model(model_info)

    for i, statement in enumerate(statements):
        print(f"Statement {i} / {num_statements} [{model_name}]:", statement)
        prompt = prompt_template.format(statement=statement)
        print("Prompt:", prompt)
        
        for j in range(10):
            try:
                start_time = time.time()
                while True:
                    try:
                        if model_family == "local":
                            response = model.generate(prompt, max_tokens=max_tokens)
                        else:
                            response = model.generate(prompt, max_tokens=max_tokens)
                        break
                    except Exception as e:
                        if time.time() - start_time > 20:
                            raise e
                        print(f"Exception: {type(e).__name__}, {str(e)}")
                        print("Retrying generation due to exception...")
                        time.sleep(2)
                    # Check if we are about to exceed the OpenAI rate limit
                    if model_family == "openai" and i % 60 == 0 and i != 0:
                        print(
                            "Sleeping for 60 seconds to avoid exceeding OpenAI rate limit..."
                        )
                        time.sleep(60)
                break
            except Exception as e:
                print(f"Exception: {type(e).__name__}, {str(e)}")
                print("Skipping API error", j)
                time.sleep(2)
        
        print("Response:", response)
        inputs.append(statement)
        responses.append(response)
        full_conversations.append(prompt + response)

    # print all variables
    print("Statements:", statements)
    print("Model family:", model_family)
    print("Model name:", model_name)
    print("Prompt template:", prompt_template)
    print("Inputs:", inputs)
    print("Responses:", responses)
    print("Full conversations:", full_conversations)

    query_results["inputs"] = inputs
    query_results["responses"] = responses
    query_results["full_conversations"] = full_conversations

    return query_results
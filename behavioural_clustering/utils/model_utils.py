import time

from behavioural_clustering.models.model_factory import initialize_model


def query_model_on_statements(
    statements, model_family, model, prompt_template, system_message
):
    query_results = {}
    inputs, responses, full_conversations = [], [], []
    model_info = {}
    model_info["model_family"] = model_family
    model_info["model"] = model
    model_info["system_message"] = system_message
    query_results["model_info"] = model_info

    print("query_model_on_statements...")

    model_instance = initialize_model(model_info)

    for i, statement in enumerate(statements):
        print(f"statement {i}:", statement)
        prompt = prompt_template.format(statement=statement)
        print("prompt:", prompt)
        for j in range(10):
            try:
                start_time = time.time()
                while True:
                    try:
                        response = model_instance.generate(prompt)

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
        print("response:", response)
        inputs.append(statement)
        responses.append(response)
        full_conversations.append(prompt + response)

    # print all variables
    print("statements:", statements)
    print("model_family:", model_family)
    print("model:", model)
    print("prompt_template:", prompt_template)
    print("inputs:", inputs)
    print("responses:", responses)
    print("full_conversations:", full_conversations)

    query_results["inputs"] = inputs
    query_results["responses"] = responses
    query_results["full_conversations"] = full_conversations

    return query_results
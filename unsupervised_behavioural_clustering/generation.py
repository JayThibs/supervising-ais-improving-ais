def query_model(model, prompts):
    return [model.generate(prompt) for prompt in prompts]

from models import MODELS, OPENAI_GPT3

models_to_evaluate = [OPENAI_GPT3]  # or MODELS

for model in models_to_evaluate:
    loaded_model = model.load()
    # evaluate model...

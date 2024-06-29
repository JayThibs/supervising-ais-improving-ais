import random
import time

from behavioural_clustering.models.model_factory import initialize_model


def identify_theme(
    texts,
    model_info,
    sampled_texts=5,
    temp=0.5,
    max_tokens=70,
    max_total_tokens=250,
    instructions="Briefly describe the overall theme of the following texts. Do not give the theme of any individual text.",
):
    theme_identify_prompt = instructions + "\n\n"
    model_info["system_message"] = ""
    sampled_texts = random.sample(texts, min(len(texts), sampled_texts))
    for i in range(len(sampled_texts)):
        theme_identify_prompt = (
            theme_identify_prompt
            + "Text "
            + str(i + 1)
            + ": "
            + str(sampled_texts[i])  # Convert to string
            + "\n"
        )
    theme_identify_prompt = theme_identify_prompt + "\nTheme:"
    model_instance = initialize_model(model_info, temp, max_tokens)
    for i in range(20):
        try:
            completion = model_instance.generate(theme_identify_prompt)[
                :max_total_tokens
            ].replace("\n", " ")
            break
        except:
            print("Skipping API error", i)
            time.sleep(2)
    return completion
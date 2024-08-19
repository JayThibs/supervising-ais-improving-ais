import time
import numpy as np
import sklearn
import pdb
import scipy
from terminaltables import AsciiTable
from typing import Tuple, Any, Optional, List
from openai import OpenAI

### Unused functions ###
def select_by_indices(curation_results, indices):
    found_points = []
    for i in indices:
        point_status = []
        point_status.append(curation_results[0][i][0])
        point_status.append([])
        for cr in curation_results:
            point_status[1].append(cr[i][1])
        found_points.append(point_status)
    return found_points

def text_match_theme(
    text,
    theme,
    matching_instructions='Does the following text contain themes of "{theme}"? Answer either "yes" or "no".\nText: "{text}"\nAnswer:',
    model_family="openai",
    model="gpt-3.5-turbo",
    system_message="You are an AI language model.",
    temperature=0,
    max_tokens=3,
):
    client = OpenAI()
    matching_instructions = matching_instructions.format(theme=theme, text=text)
    # model_instance = initialize_model(model_info, temperature, max_tokens)
    for i in range(20):
        try:
            completion = OpenAI.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": matching_instructions}],
                max_tokens=3,
                temperature=temperature,
            )
            break
        except:
            print("Skipping API error", i)
            time.sleep(2)
    return "yes" in str.lower(completion.choices[0].message.content)


def purify_cluster(cluster_texts_and_embeddings, theme_list):
    n_themes = len(theme_list)
    n_texts = len(cluster_texts_and_embeddings)
    texts = [e[0] for e in cluster_texts_and_embeddings]
    embeddings = np.array([e[1] for e in cluster_texts_and_embeddings])
    pure_cluster_texts_and_embeddings = [[] for _ in range(n_themes)]
    pure_cluster_counts = [0 for _ in range(n_themes)]
    remaining_texts_mask = np.ones(n_texts, dtype=bool)

    for i, theme in enumerate(theme_list):
        for j, (embed, text) in enumerate(zip(embeddings, texts)):
            if text_match_theme(text, theme):
                pure_cluster_texts_and_embeddings[i].append([text, embed])
                pure_cluster_counts[i] += 1
                remaining_texts_mask[j] = False
    return (
        pure_cluster_texts_and_embeddings,
        pure_cluster_counts,
        [cluster_texts_and_embeddings[i] for i in range(n_texts) if remaining_texts_mask[i]],
    )
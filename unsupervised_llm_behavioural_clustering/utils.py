import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from terminaltables import AsciiTable
import random
import time
import tqdm
import openai


def query_model(statements, model, prompt, model_name="gpt-3.5-turbo"):
    """
    Query a model with a list of statements and a prompt.
    """
    inputs = []
    responses = []
    full_conversations = []
    n_statements = len(statements)
    chain = LLMChain(llm=model, prompt=prompt)
    prompts = [prompt.format(statement=s) for s in statements]
    results = model.generate(prompts).generations
    for i in range(n_statements):
        r = results[i][0].text
        inputs.append(statements[i])
        responses.append(r)
        full_conversations.append(prompts[i] + r)
    return inputs, responses, full_conversations


def get_model_approval(
    statements, prompt, system_role_str, approve_strs=["yes"], disapprove_strs=["no"]
):
    """Get model's approval for a list of statements."""

    try:
        approvals = []
        n_statements = len(statements)
        prompts = [prompt.format(statement=s) for s in statements]

        for i in tqdm.tqdm(range(n_statements)):
            for j in range(20):
                try:
                    comp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_role_str},
                            {"role": "user", "content": prompts[i]},
                        ],
                        temperature=0,
                        max_tokens=5,
                    )
                    break
                except:
                    print("Server error", j)
                    time.sleep(2)
            r = str.lower(comp["choices"][0]["message"]["content"])

            approve_strs_in_response = sum([s in r for s in approve_strs])
            disapprove_strs_in_response = sum([s in r for s in disapprove_strs])

            if approve_strs_in_response and not disapprove_strs_in_response:
                approvals.append(1)
            elif not approve_strs_in_response and disapprove_strs_in_response:
                approvals.append(0)
            else:
                # Uncertain response:
                approvals.append(-1)
        return prompts, approvals

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_joint_embedding(
    inputs, responses, model="text-embedding-ada-002", combine_statements=False
):
    """
    Get joint embedding for a list of inputs and responses.
    """
    if not combine_statements:
        inputs_embeddings = embed_texts(inputs)
        responses_embeddings = embed_texts(responses)
        joint_embeddings = [
            i + r for i, r in zip(inputs_embeddings, responses_embeddings)
        ]
    else:
        joint_embeddings = embed_texts(
            [input + response for input, response in zip(inputs, responses)]
        )
    return joint_embeddings


def identify_theme(
    texts,
    sampled_texts=5,
    model="gpt-3.5-turbo",
    temp=1,
    max_tokens=50,
    instructions="Briefly describe the overall theme of the following texts:",
):
    """
    Summarizes key themes from a sample of texts from a cluster.
    """
    theme_identify_prompt = instructions + "\n\n"
    sampled_texts = random.sample(texts, min(len(texts), sampled_texts))
    for i in range(len(sampled_texts)):
        theme_identify_prompt = (
            theme_identify_prompt
            + "Text "
            + str(i + 1)
            + ": "
            + sampled_texts[i]
            + "\n"
        )
    for i in range(20):
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": theme_identify_prompt}],
                max_tokens=max_tokens,
                temperature=temp,
            )
            break
        except:
            print("Skipping API error", i)
            time.sleep(2)
    return completion["choices"][0]["message"]["content"]


def get_cluster_stats(joint_embeddings_all_llms, cluster_labels, cluster_ID):
    """
    Analyzes a cluster and extracts aggregated statistics.
    """
    inputs = []
    responses = []
    cluster_size = 0
    n_llms = max([e[0] for e in joint_embeddings_all_llms])
    fractions = [0 for _ in range(n_llms + 1)]
    n_datapoints = len(joint_embeddings_all_llms)
    for e, l in zip(joint_embeddings_all_llms, cluster_labels):
        if l != cluster_ID:
            continue
        if e[0] >= 0:
            fractions[e[0]] += 1
        cluster_size += 1
        inputs.append(e[1])
        responses.append(e[2])
    return inputs, responses, [f / cluster_size for f in fractions]


def get_cluster_approval_stats(
    approvals_statements_and_embeddings, cluster_labels, cluster_ID
):
    """
    Analyzes a cluster and extracts aggregated approval statistics.
    """
    inputs = []
    responses = []
    cluster_size = 0
    n_conditions = len(approvals_statements_and_embeddings[0][0])
    approval_fractions = [0 for _ in range(n_conditions)]
    n_datapoints = len(approvals_statements_and_embeddings)
    for e, l in zip(approvals_statements_and_embeddings, cluster_labels):
        if l != cluster_ID:
            continue
        for i in range(n_conditions):
            if e[0][i] == 1:
                approval_fractions[i] += 1
        cluster_size += 1
        inputs.append(e[1])
    return inputs, [f / cluster_size for f in approval_fractions]


def get_cluster_centroids(embeddings, cluster_labels):
    """
    Calculates the centroid for each cluster.
    """
    centroids = []
    for i in range(max(cluster_labels) + 1):
        c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
        centroids.append(c)
    return np.array(centroids)


def compile_cluster_table(
    clustering,
    approvals_statements_and_embeddings,
    theme_summary_instructions="Briefly describe the overall theme of the following texts:",
    max_desc_length=250,
):
    """
    Tabulates high-level statistics and themes for each cluster.
    """
    n_clusters = max(clustering.labels_) + 1
    rows = []
    for cluster_id in tqdm.tqdm(range(n_clusters)):
        row = [str(cluster_id)]
        cluster_indices = np.arange(len(clustering.labels_))[
            clustering.labels_ == cluster_id
        ]
        row.append(len(cluster_indices))
        inputs, model_approval_fractions = get_cluster_approval_stats(
            approvals_statements_and_embeddings, clustering.labels_, cluster_id
        )
        for frac in model_approval_fractions:
            row.append(str(round(100 * frac, 1)) + "%")
        cluster_inputs_themes = [
            identify_theme(
                inputs,
                sampled_texts=10,
                max_tokens=70,
                temp=0.5,
                instructions=theme_summary_instructions,
            )[:max_desc_length].replace("\n", " ")
            for _ in range(1)
        ]
        inputs_themes_str = "\n".join(cluster_inputs_themes)
        row.append(inputs_themes_str)
        rows.append(row)
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return rows

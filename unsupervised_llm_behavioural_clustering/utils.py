import numpy as np
import sklearn
import pdb
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from terminaltables import AsciiTable
import random
import time
from typing import List
from models import OpenAIModel, AnthropicModel, LocalModel
from openai import OpenAI


def query_model_on_statements(statements, model_family, model, prompt_template):
    inputs = []
    responses = []
    full_conversations = []
    model_instance = None

    print("query_model_on_statements...")

    if "openai" == model_family:
        model_instance = OpenAIModel(model)
    elif "anthropic" == model_family:
        model_instance = AnthropicModel()
    elif model_family == "local":
        model_instance = LocalModel()
    else:
        raise ValueError("Invalid model name")

    for statement in statements:
        print("statement:", statement)
        prompt = prompt_template.format(statement=statement)
        print("prompt:", prompt)
        response = model_instance.generate(prompt)
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
    print("model_instance:", model_instance)

    return inputs, responses, full_conversations, model_instance


def get_model_approvals(
    statements,
    prompt_template,
    model,
    system_role_str,
    approve_strs=["yes"],
    disapprove_strs=["no"],
):
    approvals = []
    model_instance = None

    if model == "openai":
        model_instance = OpenAIModel()
    elif model == "anthropic":
        model_instance = AnthropicModel()
    elif model == "local":
        model_instance = LocalModel()
    else:
        raise ValueError("Invalid model name")

    for statement in tqdm(statements):
        prompt = prompt_template.format(statement=statement)
        response = model_instance.generate(prompt).lower()

        is_approved = any(s in response for s in approve_strs)
        is_disapproved = any(s in response for s in disapprove_strs)

        if is_approved and not is_disapproved:
            approvals.append(1)
        elif not is_approved and is_disapproved:
            approvals.append(0)
        else:
            approvals.append(-1)

    return statements, approvals


def embed_texts(
    texts: List[List[str]],
    model="text-embedding-ada-002",
    batch_size=20,
    max_retries=50,
    initial_sleep_time=2,
):
    client = OpenAI()
    embeddings = []
    n_texts = len(texts)
    n_batches = n_texts // batch_size + int(n_texts % batch_size != 0)

    for i in tqdm(range(n_batches)):
        for retry_count in range(max_retries):
            try:
                start_idx = batch_size * i
                end_idx = min(batch_size * (i + 1), n_texts)
                print(texts)
                text_subset = texts[start_idx:end_idx]
                print("text_subset:", text_subset)
                embeddings_data = client.embeddings.create(
                    model=model, input=text_subset
                ).data

                break  # Exit the retry loop if successful
            except Exception as e:
                print(f"Skipping due to server error number {retry_count}: {e}")
                time.sleep(
                    initial_sleep_time * (2**retry_count)
                )  # Exponential backoff

        # print("embedding:", embedding)
        embeddings += [item.embedding for item in embeddings_data]
    return embeddings


def get_joint_embedding(
    inputs,
    responses,
    model_name="text-embedding-ada-002",  # Added model_name as a parameter
    combine_statements=False,
):
    """Get joint embedding for a list of inputs and responses."""
    if not combine_statements:
        inputs_embeddings = embed_texts(texts=[inputs])
        responses_embeddings = embed_texts(texts=[responses])
        joint_embeddings = [
            i + r for i, r in zip(inputs_embeddings, responses_embeddings)
        ]
    else:
        print("inputs:", inputs)
        print("responses:", responses)
        joint_embeddings = embed_texts(
            texts=[input + " " + response for input, response in zip(inputs, responses)]
        )
    return joint_embeddings


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


def identify_theme(
    texts,
    model_instance,
    sampled_texts=5,
    temp=0.1,
    max_tokens=50,
    instructions="Briefly describe the overall theme of the following texts. Do not give the theme of any individual text.",
):
    theme_identify_prompt = instructions + "\n\n"
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
    for i in range(20):
        try:
            completion = model_instance.generate(theme_identify_prompt)
            break
        except:
            print("Skipping API error", i)
            time.sleep(2)
    return completion


# Still need to test the following two functions
def text_match_theme(
    text,
    theme,
    matching_instructions='Does the following text contain themes of "{theme}"? Answer either "yes" or "no".\nText: "{text}"\nAnswer:',
    model="gpt-3.5-turbo",
):
    client = OpenAI()
    for i in range(20):
        try:
            completion = OpenAI.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": matching_instructions}],
                max_tokens=3,
                temperature=0,
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
    embeddings = [e[1] for e in cluster_texts_and_embeddings]
    pure_cluster_texts_and_embeddings = [[] for _ in range(n_themes)]
    pure_cluster_counts = [0 for _ in range(n_themes)]
    remaining_texts_mask = [True for _ in range(n_texts)]

    for i, theme in enumerate(theme_list):
        for j, (embed, text) in enumerate(zip(embeddings, texts)):
            if text_match_theme(text, theme):
                pure_cluster_texts_and_embeddings[i].append([text, embed])
                pure_cluster_counts[i] += 1
                remaining_texts_mask[j] = False
    return (
        pure_cluster_texts_and_embeddings,
        pure_cluster_counts,
        cluster_texts_and_embeddings[remaining_texts_mask],
    )


def lookup_cid_pos_in_rows(rows, cid):
    for i in range(len(rows)):
        if int(rows[i][0]) == cid:
            return i
    return -1


def print_cluster(cid, labels, joint_embeddings_all_llms, rows=None):
    print(
        "####################################################################################"
    )
    if not rows is None:
        cid_pos = lookup_cid_pos_in_rows(rows, cid)
        print("### Input desc:")
        print(rows[cid_pos][4])
        print("### Response desc:")
        print(rows[cid_pos][5])
        print("### Interaction desc:")
        print(rows[cid_pos][6])
        print(rows[cid_pos][2] + " / " + rows[cid_pos][3])
    for i in range(len(labels)):
        if labels[i] == cid:
            print(
                "============================================================\nPoint "
                + str(i)
                + ": ("
                + str(2 + joint_embeddings_all_llms[i][0])
                + ")"
            )
            print(joint_embeddings_all_llms[i][1])
            print(joint_embeddings_all_llms[i][2])


def print_cluster_approvals(
    cid, labels, approvals_statements_and_embeddings, rows=None
):
    print(
        "####################################################################################"
    )
    if not rows is None:
        cid_pos = lookup_cid_pos_in_rows(rows, cid)
        print("### Input desc:")
        print(rows[cid_pos][-1])
        print(rows[cid_pos][1:-1])
    for i in range(len(labels)):
        if labels[i] == cid:
            print(
                "============================================================\nPoint "
                + str(i)
                + ":"
            )
            print(approvals_statements_and_embeddings[i][1])


def get_cluster_stats(joint_embeddings_all_llms, cluster_labels, cluster_ID):
    """Analyzes a cluster and extracts aggregated statistics."""
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
    """Analyzes a cluster and extracts aggregated approval statistics."""
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
    """Calculates the centroid for each cluster."""
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
    """Tabulates high-level statistics and themes for each cluster."""
    n_clusters = max(clustering.labels_) + 1
    rows = []
    for cluster_id in tqdm(range(n_clusters)):
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


def compare_response_pair(
    approvals_statements_and_embeddings, r_1_name, r_2_name, labels, response_type
):
    response_type_int = lookup_response_type_int(response_type)
    r_1_index = lookup_name_index(labels, r_1_name)
    r_2_index = lookup_name_index(labels, r_2_name)

    r_1_mask = np.array(
        [
            e[0][r_1_index] == response_type_int
            for e in approvals_statements_and_embeddings
        ]
    )
    r_2_mask = np.array(
        [
            e[0][r_2_index] == response_type_int
            for e in approvals_statements_and_embeddings
        ]
    )

    print(r_1_name + ' "' + response_type + '" responses:', sum(r_1_mask))
    print(r_2_name + ' "' + response_type + '" responses:', sum(r_2_mask))

    print(
        "Intersection matrix for "
        + r_1_name
        + " and "
        + r_2_name
        + ' "'
        + response_type
        + '" responses:'
    )
    conf_matrix = sklearn.metrics.confusion_matrix(r_1_mask, r_2_mask)
    conf_rows = [["", "Not In " + r_1_name, "In " + r_1_name]]
    conf_rows.append(["Not In " + r_2_name, conf_matrix[0][0], conf_matrix[1][0]])
    conf_rows.append(["In " + r_2_name, conf_matrix[0][1], conf_matrix[1][1]])
    t = AsciiTable(conf_rows)
    t.inner_row_border = True
    print(t.table)

    pearson_r = scipy.stats.pearsonr(r_1_mask, r_2_mask)
    print(
        'Pearson correlation between "'
        + response_type
        + '" responses for '
        + r_1_name
        + " and "
        + r_2_name
        + ":",
        round(pearson_r.correlation, 5),
    )
    print("(P-value " + str(round(pearson_r.pvalue, 5)) + ")")


def lookup_name_index(labels, name):
    for i, l in zip(range(len(labels)), labels):
        if name == l:
            return i
    print("Please provide valid name")
    return


def lookup_response_type_int(response_type):
    response_type = str.lower(response_type)
    if response_type in ["approve", "approval", "a"]:
        return 1
    elif response_type in ["disapprove", "disapproval", "d"]:
        return 0
    elif response_type in ["no response", "no decision", "nr", "nd"]:
        return -1
    print("Please provide valid response type.")
    return

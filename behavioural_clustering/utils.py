import os
import time
import random
import pickle
import logging
import datetime
import numpy as np
import sklearn
import torch
import pdb
from tqdm import tqdm
import scipy
from terminaltables import AsciiTable
from typing import Tuple, Any, Optional, List
from openai import OpenAI
from models import initialize_model


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


# Still need to test the following two functions
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

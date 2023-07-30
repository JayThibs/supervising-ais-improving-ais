def query_model_on_statements(statements, llm, prompt):
    # Initialize empty lists for each output variable
    inputs = []
    responses = []
    full_conversations = []
    # Get the number of statements
    n_statements = len(statements)
    # Create a chain object
    chain = LLMChain(llm=llm, prompt=prompt)
    # Create a list of prompts, one for each statement
    prompts = [prompt.format(statement=s) for s in statements]
    # Generate responses to each prompt
    results = llm.generate(prompts).generations
    # Iterate through each statement
    for i in range(n_statements):
        # Get the response
        r = results[i][0].text
        # Add the statement and response to the appropriate lists
        inputs.append(statements[i])
        responses.append(r)
        full_conversations.append(prompts[i] + r)
    return inputs, responses, full_conversations


def ask_model_approve_statements(
    statements, prompt, system_role_str, approve_strs=["yes"], disapprove_strs=["no"]
):
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
        # print("SYSTEM    :", system_role_str)
        # print("USER      :", prompts[i])
        # print("RESPONSE  :", r)

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


def embed_texts(texts, model="text-embedding-ada-002"):
    embeddings = []
    n_texts = len(texts)
    n_batches = n_texts // 20 + int(n_texts % 20 != 0)
    for i in tqdm.tqdm(range(n_batches)):
        for j in range(50):
            try:
                texts_subset = texts[20 * i : min(20 * (i + 1), n_texts)]
                embedding = openai.Embedding.create(
                    model="text-embedding-ada-002", input=texts_subset
                )
                break
            except:
                print("Skipping server error number " + str(j))
                time.sleep(2)
        embeddings += [e["embedding"] for e in embedding["data"]]
    return embeddings


def joint_embedding(
    inputs, responses, model="text-embedding-ada-002", combine_statements=False
):
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
    sampled_texts=5,
    model="gpt-3.5-turbo",
    temp=1,
    max_tokens=50,
    instructions="Briefly describe the overall theme of the following texts:",
):
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
    # theme_identify_prompt = theme_identify_prompt + "\nTheme:"
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


# Still need to test the following two functions
def text_match_theme(
    text,
    theme,
    matching_instructions='Does the following text contain themes of "{theme}"? Answer either "yes" or "no".\nText: "{text}"\nAnswer:',
    model="gpt-3.5-turbo",
):
    for i in range(20):
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": matching_instructions}],
                max_tokens=3,
                temperature=0,
            )
            break
        except:
            print("Skipping API error", i)
            time.sleep(2)
    return "yes" in str.lower(completion["choices"][0]["message"]["content"])


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


def print_cluster(cid, labels, rows=None):
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
    centroids = []
    for i in range(max(cluster_labels) + 1):
        c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
        centroids.append(c)
    return np.array(centroids)


def plot_approvals(
    dim_reduce,
    approvals_statements_and_embeddings,
    response_type_int,
    colors,
    shapes,
    labels,
    sizes,
    title,
    order=None,
):
    n_persona = len(labels)
    if order is None:
        order = [i for i in range(n_persona)]

    masks = [
        np.array(
            [e[0][i] == response_type_int for e in approvals_statements_and_embeddings]
        )
        for i in range(n_persona)
    ]
    plt.scatter(
        dim_reduce[:, 0], dim_reduce[:, 1], c="grey", label="None", s=10, alpha=0.5
    )
    for i in order:
        plt.scatter(
            dim_reduce[:, 0][masks[i]],
            dim_reduce[:, 1][masks[i]],
            marker=shapes[i],
            c=colors[i],
            label=labels[i],
            s=sizes[i],
        )
    plt.title(title)
    plt.legend()
    plt.show()


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


def hierarchical_cluster(
    clustering,
    approvals_statements_and_embeddings,
    rows,
    colors,
    labels=None,
    bar_height=1,
    bb_width=10,
    x_leftshift=0,
    y_downshift=0,
    figsize=(35, 35),
    filename="hierarchical_clustering.pdf",
):
    def llf(id):
        if id < n_clusters:
            return leaf_labels[id]
        else:
            return "Error: id too high."

    statement_embeddings = np.array([e[2] for e in approvals_statements_and_embeddings])
    centroids = get_cluster_centroids(statement_embeddings, clustering.labels_)
    Z = linkage(centroids, "ward")

    n_clusters = max(clustering.labels_) + 1
    cluster_labels = []
    for i in range(n_clusters):
        pos = lookup_cid_pos_in_rows(rows, i)
        if pos >= 0:
            cluster_labels.append(rows[pos][-1])
        else:
            cluster_labels.append("(Label missing)")

    all_cluster_sizes = []
    for i in range(n_clusters):
        inputs, model_approval_fractions = get_cluster_approval_stats(
            approvals_statements_and_embeddings, clustering.labels_, i
        )
        n = len(inputs)
        all_cluster_sizes.append([n] + [int(f * n) for f in model_approval_fractions])

    for merge in Z:
        m1 = int(merge[0])
        m2 = int(merge[1])
        m1_sizes = all_cluster_sizes[m1]
        m2_sizes = all_cluster_sizes[m2]
        merged_sizes = [
            int(m1_entry + m2_entry) for m1_entry, m2_entry in zip(m1_sizes, m2_sizes)
        ]
        all_cluster_sizes.append(merged_sizes)

    original_cluster_sizes = all_cluster_sizes[:n_clusters]
    merged_cluster_sizes = all_cluster_sizes[n_clusters:]

    # leaf_labels=[":".join([str(condition_size) for condition_size in s]) + " : " + l for s,l in zip(original_cluster_sizes,cluster_labels)]
    leaf_labels = [
        str(s[0]) + " : " + l for s, l in zip(original_cluster_sizes, cluster_labels)
    ]

    # adapted from: https://stackoverflow.com/questions/30317688/annotating-dendrogram-nodes-in-scipy-matplotlib

    Z[:, 2] = np.arange(1.0, len(Z) + 1)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
    dn = dendrogram(Z, ax=ax, leaf_rotation=-90, leaf_font_size=20, leaf_label_func=llf)

    ii = np.argsort(np.array(dn["dcoord"])[:, 1])
    for j, (icoord, dcoord) in enumerate(zip(dn["icoord"], dn["dcoord"])):
        x = 0.5 * sum(icoord[1:3])
        y = dcoord[1]
        ind = np.nonzero(ii == j)[0][0]
        s = merged_cluster_sizes[ind]
        # ax.annotate(merged_cluster_labels[ind], (x,y), va='top', ha='center')
        for i in range(len(colors)):
            ax.add_patch(
                Rectangle(
                    (
                        x - bb_width / 2 - x_leftshift,
                        y
                        - y_downshift
                        - i * bar_height
                        + bar_height * (len(colors) - 1),
                    ),
                    bb_width * s[i + 1] / s[0],
                    bar_height,
                    facecolor=colors[i],
                )
            )

        ax.add_patch(
            Rectangle(
                (x - bb_width / 2 - x_leftshift, y - y_downshift),
                bb_width,
                bar_height * len(colors),
                facecolor="none",
                ec="k",
                lw=1,
            )
        )
    if not labels is None:
        patch_colors = [
            mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)
        ]
        ax.legend(handles=patch_colors)

    plt.tight_layout()
    plt.savefig(filename)


def compile_cluster_table(
    clustering,
    approvals_statements_and_embeddings,
    theme_summary_instructions="Briefly list the common themes of the following texts:",
    max_desc_length=250,
):
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

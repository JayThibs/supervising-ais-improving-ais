import pandas as pd
import tqdm
from treelib import Tree
import random


def add_tree_level(df):
    """helper function to add tree level to a df"""

    # if tree level already exists, return df
    if "tree_level" in df.columns:
        return df

    else:
        tree_level_map = {}

        # iterate over rows in df
        for i, row in df.iterrows():
            message_id = row["message_id"]
            parent_id = row["parent_id"]

            # if parent_id is None, then it is a root message
            if parent_id is None:
                tree_level_map[message_id] = 0
            # if parent_id is the same as message_tree_id, then it is a direct reply to the root message
            elif parent_id == row["message_tree_id"]:
                tree_level_map[message_id] = 1
            # else just look up the tree level of the parent_id and add 1
            else:
                tree_level_map[message_id] = tree_level_map[parent_id] + 1

        # create a df from the tree_level_map and merge it with the original df
        df_tree_level_map = (
            pd.DataFrame.from_dict(tree_level_map, orient="index", columns=["tree_level"])
            .reset_index()
            .rename(columns={"index": "message_id"})
        )

        return df.merge(df_tree_level_map, on="message_id")


def get_tree_paths(df, message_tree_id, max_char_len = 2000):
    # look at all data for this message tree
    df_message_tree = df.query(f"message_tree_id == '{message_tree_id}'").sort_values("created_date")

    # add tree level to df
    df_message_tree = add_tree_level(df_message_tree)
    # lets create a tree of message texts
    text_tree = Tree()
    # lets set a max char length for the text

    # iterate over rows in df_message_tree
    for i, row in df_message_tree.iterrows():
        # grab the message_id, parent_id, text, and parent text
        message_id = row["message_id"]
        parent_id = row["parent_id"]
        text = row["text"]
        text_short = text[:max_char_len] if len(text) > max_char_len else text
        #text_short = text_short.encode(encoding="ascii", errors="replace").decode()
        text_short = text_short.replace("\n", " ")
        parent_text = (
            df_message_tree.query(f"message_id == '{parent_id}'")["text"].values[0] if parent_id is not None else "ROOT"
        )
        parent_text_short = parent_text[:max_char_len] if len(parent_text) > max_char_len else parent_text
        #parent_text_short = parent_text_short.encode(encoding="ascii", errors="replace").decode()
        parent_text_short = parent_text_short.replace("\n", " ")

        # if parent_id is None, then it is a root message so dont add parent text as is none
        if parent_id is None:
            text_tree.create_node(text_short, text_short)
        # else use the parent text short as the parent
        else:
            try:
                text_tree.create_node(text_short, text_short, parent=parent_text_short)
            except:
                pass
    return text_tree.paths_to_leaves()



def extract_convo_trees(df):
    message_tree_ids = df["message_tree_id"].drop_duplicates().values
    all_conversations = []
    all_convo_texts = []

    for id in tqdm.tqdm(message_tree_ids):
        convos = get_tree_paths(df, id, max_char_len = 2000)
        for c in convos:
            all_conversations.append(c)
            text = ""
            for i in range(len(c)):
                if i % 2 == 0:
                    text = text + "USER: " + c[i]
                else:
                    text = text + "ASSISTANT: " + c[i]
                if i < len(c) - 1:
                    text = text + "\n"
            all_convo_texts.append(text)
    random.shuffle(all_convo_texts)
    return all_convo_texts

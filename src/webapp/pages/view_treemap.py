import streamlit as st
import plotly.graph_objs as go
import textwrap
import json
from pathlib import Path

def parse_hierarchical_data(hierarchical_data, prompt_labels):
    linkage_matrix, descriptions, original_counts, merged_counts, n_clusters = hierarchical_data

    tree_dict = {}

    for i, (desc, counts) in enumerate(zip(descriptions, original_counts)):
        tree_dict[i] = {
            "name": f"Cluster_{i}",
            "description": desc,
            "value": counts[0],
            "proportions": {label: count / counts[0] for label, count in zip(prompt_labels, counts[1:])}
        }

    for i, link in enumerate(linkage_matrix):
        left, right, _, _ = link
        node_id = i + n_clusters
        left_child = tree_dict[int(left)]
        right_child = tree_dict[int(right)]

        tree_dict[node_id] = {
            "name": f"Cluster_{node_id}",
            "children": [left_child, right_child],
            "value": left_child["value"] + right_child["value"],
            "proportions": {
                k: (left_child["value"] * left_child["proportions"][k] + right_child["value"] * right_child["proportions"][k])
                / (left_child["value"] + right_child["value"])
                for k in left_child["proportions"]
            },
        }

    return tree_dict[node_id]

def format_tree(node, parent_name="", level=0):
    results = []
    node_name = node["name"]
    results.append({
        "name": node_name,
        "parent": parent_name,
        "value": node["value"],
        "level": level,
        "proportions": node["proportions"],
        "description": node.get("description", ""),
    })

    if "children" in node:
        for child in node["children"]:
            results.extend(format_tree(child, node_name, level + 1))

    return results

def create_treemap(df, max_level, color_by):
    fig = go.Figure(go.Treemap(
        labels=[item["name"] for item in df if item["level"] <= max_level],
        parents=[item["parent"] for item in df if item["level"] <= max_level],
        values=[item["value"] for item in df if item["level"] <= max_level],
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Proportions:<br>%{customdata}<extra></extra>",
        customdata=[
            "<br>".join([f"{k}: {v:.2f}" for k, v in item["proportions"].items()])
            + ("<br><br>" + "<br>".join(textwrap.wrap(item["description"], width=50)) if item["description"] else "")
            for item in df if item["level"] <= max_level
        ],
        marker=dict(
            colors=[item["proportions"][color_by] for item in df if item["level"] <= max_level],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=f"{color_by} Proportion"),
        ),
        texttemplate="<b>%{label}</b><br>%{text}",
        text=[
            "<br>".join([f"{k}: {v:.2f}" for k, v in item["proportions"].items()])
            for item in df if item["level"] <= max_level
        ],
        textposition="middle center",
    ))

    fig.update_layout(
        title_text="Hierarchical Clustering Treemap",
        width=1000,
        height=800,
    )

    return fig

def show():
    st.title("Interactive Treemap Visualization")

    data_accessor = st.session_state.data_accessor
    run_ids = data_accessor.get_available_run_ids()
    selected_run_id = st.selectbox("Select Run ID", run_ids, key="run_id")

    prompt_types = data_accessor.get_available_prompt_types(selected_run_id)
    selected_prompt_type = st.selectbox("Select Prompt Type", prompt_types, key="prompt_type")

    # Try different naming conventions for the hierarchical clustering data
    possible_data_types = [
        f"hierarchical_clustering_{selected_prompt_type}",
        f"hierarchical_clustering_personas",
        "hierarchical_clustering"
    ]

    hierarchical_data = None
    for data_type in possible_data_types:
        try:
            hierarchical_data = data_accessor.get_run_data(selected_run_id, data_type)
            if hierarchical_data is not None:
                break
        except ValueError:
            continue

    if hierarchical_data is not None:
        prompt_category = selected_prompt_type.split("_")[-1]
        prompt_labels = data_accessor.get_prompt_labels(selected_run_id, prompt_category)
        model_names = data_accessor.get_model_names(selected_run_id)

        if isinstance(hierarchical_data, dict):
            # Multiple models
            tree_data = {}
            for model_name, model_data in hierarchical_data.items():
                root = parse_hierarchical_data(model_data, prompt_labels)
                tree_data[model_name] = format_tree(root)
        else:
            # Single model
            root = parse_hierarchical_data(hierarchical_data, prompt_labels)
            tree_data = {model_names[0]: format_tree(root)}

        # Create Streamlit widgets
        model_select = st.selectbox("Select Model", list(tree_data.keys()), key="model")
        max_level = max(item["level"] for item in tree_data[model_select])
        level_slider = st.slider("Max Level", min_value=1, max_value=max_level, value=2, key="level")
        color_by = st.selectbox("Color by", prompt_labels, key="color")

        # Create and display the treemap
        fig = create_treemap(tree_data[model_select], level_slider, color_by)
        st.plotly_chart(fig)

    else:
        st.error("Failed to load hierarchical data. Please check the data files and try again.")

if __name__ == "__main__":
    show()
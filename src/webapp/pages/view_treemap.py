import pickle
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

def load_treemap_data(run_id, prompt_type):
    data_accessor = st.session_state.data_accessor
    try:
        # Extract the last part of the prompt_type (e.g., "awareness" from "approvals_statements_awareness")
        prompt_type_suffix = prompt_type.split("_")[-1]
        file_path = data_accessor.get_file_path(run_id, f"hierarchical_clustering_{prompt_type_suffix}")
        print(f"Attempting to load hierarchical data from: {file_path}")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded hierarchical data for run {run_id} and prompt type {prompt_type}")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    
    return None

def create_treemap_data(hierarchical_data):
    treemap_data = {
        "model_names": [],
        "max_levels": {},
        "tree_data": {}
    }

    for model_name, model_data in hierarchical_data.items():
        treemap_data["model_names"].append(model_name)
        treemap_data["tree_data"][model_name] = []

        linkage_matrix, leaf_labels, original_cluster_sizes, merged_cluster_sizes, n_clusters = model_data

        # Add a single root node
        root_node = {
            "name": "Root",
            "parent": "",
            "value": sum(original_cluster_sizes[i][0] for i in range(len(original_cluster_sizes))),
            "level": 0,
            "proportions": {label: 0 for label in original_cluster_sizes[0][1:]},
            "description": "Root node"
        }
        treemap_data["tree_data"][model_name].append(root_node)

        # Add leaf nodes
        for i, (label, sizes) in enumerate(zip(leaf_labels, original_cluster_sizes)):
            treemap_data["tree_data"][model_name].append({
                "name": f"Cluster_{i}",
                "parent": "Root",
                "value": sizes[0],
                "level": 1,
                "proportions": {label: size / sizes[0] for label, size in zip(root_node["proportions"].keys(), sizes[1:])},
                "description": label
            })

        # Add merged nodes
        for i, (left, right, _, _) in enumerate(linkage_matrix):
            node_id = i + n_clusters
            left_child = treemap_data["tree_data"][model_name][int(left) + 1]
            right_child = treemap_data["tree_data"][model_name][int(right) + 1]
            
            merged_sizes = merged_cluster_sizes[node_id - n_clusters]
            treemap_data["tree_data"][model_name].append({
                "name": f"Cluster_{node_id}",
                "parent": "Root",
                "value": merged_sizes[0],
                "level": max(left_child["level"], right_child["level"]) + 1,
                "proportions": {label: size / merged_sizes[0] for label, size in zip(root_node["proportions"].keys(), merged_sizes[1:])},
                "description": ""
            })
            
            # Update parents of child nodes
            left_child["parent"] = f"Cluster_{node_id}"
            right_child["parent"] = f"Cluster_{node_id}"

        treemap_data["max_levels"][model_name] = int(linkage_matrix[-1, 2]) + 1

    return treemap_data

def create_treemap(df, max_level, color_by, model_name, plot_type):
    labels = []
    parents = []
    values = []
    colors = []

    for item in df:
        if item["level"] <= max_level and item["parent"]:
            labels.append(item["name"])
            parents.append(item["parent"])
            values.append(item["value"])
            colors.append(item["proportions"][color_by])

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=f"{color_by} Proportion"),
        ),
    ))

    fig.update_layout(
        title_text=f"Hierarchical Clustering Treemap - {plot_type} - {model_name}",
        width=1000,
        height=800,
    )

    return fig

def show():
    st.title("Interactive Treemap Visualization")

    data_accessor = st.session_state.data_accessor
    run_ids = data_accessor.get_available_run_ids()
    selected_run_id = st.selectbox("Select Run ID", run_ids)

    prompt_types = data_accessor.get_available_prompt_types(selected_run_id)
    selected_prompt_type = st.selectbox("Select Prompt Type", prompt_types)

    hierarchical_data = load_treemap_data(selected_run_id, selected_prompt_type)

    if hierarchical_data:
        treemap_data = create_treemap_data(hierarchical_data)

        st.sidebar.header("Treemap Controls")
        model_name = st.sidebar.selectbox("Select Model", treemap_data["model_names"])
        max_level = st.sidebar.slider("Max Level", 1, treemap_data["max_levels"][model_name], 2)
        
        # Get the available labels from the first cluster's proportions
        available_labels = list(treemap_data["tree_data"][model_name][0]["proportions"].keys())
        color_by = st.sidebar.selectbox("Color by", available_labels)

        fig = create_treemap(
            treemap_data["tree_data"][model_name],
            max_level,
            color_by,
            model_name,
            selected_prompt_type
        )

        st.plotly_chart(fig, use_container_width=True)

        # Test with a simple treemap
        st.subheader("Test Simple Treemap")
        test_fig = go.Figure(go.Treemap(
            labels=["A", "B", "C", "D", "E", "F"],
            parents=["", "A", "A", "C", "C", "A"]
        ))
        st.plotly_chart(test_fig, use_container_width=True)
    else:
        st.warning("No hierarchical data available for the selected run and prompt type.")
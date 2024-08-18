import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pickle
import textwrap
import json
import os

def load_approval_prompts():
    with open(f"{os.getcwd()}/data/prompts/approval_prompts.json", "r") as file:
        return json.load(file)

def extract_main_theme(description):
    # Only take the first line of the description
    first_line = description.split('\n')[0]
    parts = first_line.split(':')
    if len(parts) > 1:
        return ':'.join(parts[1:]).strip()
    return first_line.strip()

def parse_hierarchical_data(hierarchical_data, labels):
    linkage_matrix, descriptions, original_counts, merged_counts, n_clusters = hierarchical_data
    
    tree_dict = {}
    
    # Create leaf nodes
    for i, (desc, counts) in enumerate(zip(descriptions, original_counts)):
        tree_dict[i] = {
            "name": f"Cluster_{i}",
            "description": desc,
            "value": counts[0],
            "proportions": {label: count / counts[0] for label, count in zip(labels, counts[1:])},
            "is_leaf": True
        }
    
    # Create non-leaf nodes
    for i, link in enumerate(linkage_matrix):
        left, right, _, _ = link
        node_id = i + n_clusters
        left_child = tree_dict[int(left)]
        right_child = tree_dict[int(right)]
        
        # Create description only for immediate children
        combined_desc = f"{left_child['name']}: {extract_main_theme(left_child['description'])}\n\n{right_child['name']}: {extract_main_theme(right_child['description'])}"
        
        tree_dict[node_id] = {
            "name": f"Cluster_{node_id}",
            "description": combined_desc,
            "children": [left_child, right_child],
            "value": left_child["value"] + right_child["value"],
            "proportions": {
                k: (left_child["value"] * left_child["proportions"][k] + 
                    right_child["value"] * right_child["proportions"][k]) / 
                   (left_child["value"] + right_child["value"])
                for k in left_child["proportions"]
            },
            "is_leaf": False
        }
    
    return tree_dict[node_id]

def format_tree(node, parent_name="", level=0):
    results = []
    node_name = node['name']
    results.append({
        'name': node_name,
        'parent': parent_name,
        'value': node['value'],
        'level': level,
        'proportions': node['proportions'],
        'description': node['description'],
        'is_leaf': node['is_leaf']
    })
    
    if 'children' in node:
        for child in node['children']:
            results.extend(format_tree(child, node_name, level + 1))
    
    return results

def create_treemap(df, max_level, color_by):
    def format_tooltip(item):
        proportions = '<br>'.join([f"{k}: {v:.2f}" for k, v in item['proportions'].items()])
        wrapped_desc = '<br>'.join(textwrap.wrap(item['description'].replace('\n\n', '<br><br>'), width=100))
        return f"<b>{item['name']}</b><br><br>{proportions}<br><br>Main theme:<br>{wrapped_desc}"

    def format_box_text(item):
        proportions = '<br>'.join([f"{k}: {v:.2f}" for k, v in item['proportions'].items()])
        wrapped_desc = '<br>'.join(textwrap.wrap(item['description'].replace('\n\n', '<br><br>'), width=50))
        return f"{item['name']}<br>{proportions}<br>Main theme: {wrapped_desc}"

    fig = go.Figure(go.Treemap(
        labels=[item['name'] for item in df if item['level'] <= max_level],
        parents=[item['parent'] for item in df if item['level'] <= max_level],
        values=[item['value'] for item in df if item['level'] <= max_level],
        branchvalues="total",
        hovertemplate='%{customdata}<extra></extra>',
        customdata=[format_tooltip(item) for item in df if item['level'] <= max_level],
        marker=dict(
            colors=[item['proportions'][color_by] for item in df if item['level'] <= max_level],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f"{color_by} Proportion")
        ),
        texttemplate='%{text}',
        text=[format_box_text(item) for item in df if item['level'] <= max_level],
        textposition='middle center',
    ))
    
    fig.update_layout(
        title_text='Hierarchical Clustering Treemap',
        width=1000,
        height=800,
    )
    
    return fig

@st.cache_data
def load_data(data_type):
    with open(f"data/results/pickle_files/hierarchy_approval_data_{data_type}.pkl", "rb") as f:
        return pickle.load(f)

def main():
    st.title('Interactive Hierarchical Clustering Treemap')

    approval_prompts = load_approval_prompts()
    data_types = list(approval_prompts.keys())

    st.sidebar.header('Treemap Controls')
    selected_data_type = st.sidebar.selectbox('Select Data Type', options=data_types)

    labels = list(approval_prompts[selected_data_type].keys())

    hierarchical_data = load_data(selected_data_type)
    root = parse_hierarchical_data(hierarchical_data, labels)
    tree_data = format_tree(root)

    max_level = max(item['level'] for item in tree_data)
    max_level = st.sidebar.slider('Max Level', min_value=1, max_value=max_level, value=max_level, step=1)
    color_by = st.sidebar.selectbox('Color by', options=labels, index=2)

    # Create a container for the plot
    plot_container = st.empty()

    # Update the plot in the container
    with plot_container.container():
        fig = create_treemap(tree_data, max_level, color_by)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
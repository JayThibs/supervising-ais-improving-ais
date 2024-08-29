import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path

def load_treemap_data(run_id, prompt_type):
    base_dir = Path(__file__).resolve().parents[3] / "data" / "results" / "viz"
    file_path = base_dir / f"interactive_treemap_{prompt_type}_{run_id}.json"
    
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        st.error(f"Treemap data not found for run {run_id} and prompt type {prompt_type}")
        return None

def create_treemap(df, max_level, color_by, model_name, plot_type):
    fig = go.Figure(go.Treemap(
        labels=[item["name"] for item in df if item["level"] <= max_level],
        parents=[item["parent"] for item in df if item["level"] <= max_level],
        values=[item["value"] for item in df if item["level"] <= max_level],
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Proportions:<br>%{customdata}<extra></extra>",
        customdata=[
            "<br>".join([f"{k}: {v:.2f}" for k, v in item["proportions"].items()])
            + ("<br><br>" + item["description"] if item["description"] else "")
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
        title_text=f"Hierarchical Clustering Treemap - {plot_type} - {model_name}",
        width=1000,
        height=800,
    )

    return fig

def show():
    st.title("Interactive Treemap Visualization")

    # Get available run IDs
    run_ids = st.session_state.data_accessor.get_available_run_ids()
    
    # Select run ID
    selected_run_id = st.selectbox("Select Run ID", run_ids)

    # Get available prompt types for the selected run
    prompt_types = st.session_state.data_accessor.get_available_prompt_types(selected_run_id)

    # Select prompt type
    selected_prompt_type = st.selectbox("Select Prompt Type", prompt_types)

    # Load treemap data
    treemap_data = load_treemap_data(selected_run_id, selected_prompt_type)

    if treemap_data:
        # Create sidebar controls
        st.sidebar.header("Treemap Controls")
        model_name = st.sidebar.selectbox("Select Model", treemap_data["model_names"])
        max_level = st.sidebar.slider("Max Level", 1, treemap_data["max_levels"][model_name], 2)
        color_by = st.sidebar.selectbox("Color by", ["Unaware", "Other AI", "Aware", "Human"])

        # Create and display the treemap
        fig = create_treemap(
            treemap_data["tree_data"][model_name],
            max_level,
            color_by,
            model_name,
            treemap_data["plot_type"]
        )
        st.plotly_chart(fig)
    else:
        st.warning("No treemap data available for the selected run and prompt type.")
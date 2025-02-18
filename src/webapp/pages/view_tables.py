import streamlit as st
import pandas as pd
from pathlib import Path
import pickle

def list_table_files(data_accessor, run_id):
    """List all table files for a specific run ID in the saved_data directory."""
    table_files = []
    if run_id in data_accessor.run_metadata:
        for data_type in data_accessor.run_metadata[run_id]["data_file_ids"].keys():
            if data_type.endswith("_table") or data_type == "compile_cluster_table":
                table_files.append(f"{run_id}_{data_type}")
    return table_files

def load_table(file_path: Path) -> pd.DataFrame:
    """Load a pickle file into a pandas DataFrame."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def show():
    st.title("Table Viewer")

    data_accessor = st.session_state.data_accessor
    
    # Get available run IDs
    available_run_ids = data_accessor.get_available_run_ids()

    if not available_run_ids:
        st.warning("No run IDs found in the metadata.")
        return

    # Allow user to select a run ID
    selected_run_id = st.selectbox("Select a run ID:", available_run_ids)

    # Get available table files for the selected run ID
    table_files = list_table_files(data_accessor, selected_run_id)

    if not table_files:
        st.warning(f"No table files found for run ID: {selected_run_id}")
        return

    selected_table = st.selectbox("Select a table to view:", table_files)

    if selected_table:
        try:
            run_id, data_type = selected_table.split('_', 1)
            file_path = data_accessor.get_file_path(run_id, data_type)
            df = load_table(file_path)

            st.subheader(f"Displaying table: {selected_table}")

            # Display basic information about the DataFrame
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {', '.join(df.columns)}")

            # Allow user to select columns to display
            selected_columns = st.multiselect(
                "Select columns to display:",
                options=list(df.columns),
                default=list(df.columns)
            )

            # Allow user to filter rows
            num_rows = st.slider("Number of rows to display:", min_value=1, max_value=len(df), value=min(10, len(df)))

            # Display the selected portion of the DataFrame
            st.dataframe(df[selected_columns].head(num_rows))

            # Option to download the full CSV
            st.download_button(
                label="Download full CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_table}.csv",
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(f"Available run IDs: {', '.join(available_run_ids)}")
            st.error(f"Selected table: {selected_table}")
            st.error(f"Run ID: {run_id}, Data type: {data_type}")
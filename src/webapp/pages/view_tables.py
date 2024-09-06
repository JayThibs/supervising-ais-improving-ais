import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List

def list_table_files(data_accessor):
    """List all CSV files in the saved_data directory."""
    csv_files = []
    for run_id in data_accessor.run_metadata.keys():
        for data_type in data_accessor.run_metadata[run_id]["data_file_ids"].keys():
            if data_type.endswith("_table"):
                csv_files.append(f"{run_id}_{data_type}")
    return csv_files

def load_table(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def show():
    st.title("Table Viewer")

    data_accessor = st.session_state.data_accessor
    
    table_files = list_table_files(data_accessor)

    if not table_files:
        st.warning("No table files found")
        return

    selected_table = st.selectbox("Select a table to view:", table_files)

    if selected_table:
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
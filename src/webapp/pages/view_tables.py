import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List

def list_table_files(directory: Path) -> List[str]:
    """List all CSV files in the given directory."""
    return [f.name for f in directory.glob('*.csv')]

def load_table(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def show():
    st.title("Table Viewer")

    # Assuming the tables are stored in the data/results/tables directory
    tables_dir = Path("data/results/tables")

    if not tables_dir.exists():
        st.error(f"Error: {tables_dir} does not exist.")
        return

    table_files = list_table_files(tables_dir)

    if not table_files:
        st.warning(f"No CSV files found in {tables_dir}")
        return

    selected_table = st.selectbox("Select a table to view:", table_files)

    if selected_table:
        file_path = tables_dir / selected_table
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
            file_name=selected_table,
            mime='text/csv',
        )
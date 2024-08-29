import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate

def load_table(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def list_table_files(directory: str) -> List[str]:
    """List all CSV files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith('.csv')]

def display_table(df: pd.DataFrame, max_rows: Optional[int] = None, max_cols: Optional[int] = None):
    """Display a pandas DataFrame with optional row and column limits."""
    if max_rows:
        df = df.head(max_rows)
    if max_cols:
        df = df.iloc[:, :max_cols]
    
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

def view_tables(directory: str):
    """View all CSV tables in the given directory."""
    table_files = list_table_files(directory)
    
    if not table_files:
        print(f"No CSV files found in {directory}")
        return

    while True:
        print("\nAvailable tables:")
        for i, file in enumerate(table_files):
            print(f"{i+1}. {file}")
        
        choice = input("\nEnter the number of the table you want to view (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(table_files):
                file_path = os.path.join(directory, table_files[index])
                df = load_table(file_path)
                print(f"\nDisplaying table: {table_files[index]}")
                
                while True:
                    max_rows = input("Enter the number of rows to display (or 'all' for all rows): ")
                    max_rows = None if max_rows.lower() == 'all' else int(max_rows)
                    
                    max_cols = input("Enter the number of columns to display (or 'all' for all columns): ")
                    max_cols = None if max_cols.lower() == 'all' else int(max_cols)
                    
                    display_table(df, max_rows, max_cols)
                    
                    action = input("\nEnter 'r' to resize, 'b' to go back to table selection, or 'q' to quit: ")
                    if action.lower() == 'b':
                        break
                    elif action.lower() == 'q':
                        return
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

if __name__ == "__main__":
    # Assuming the script is run from the project root directory
    tables_dir = Path("data/results/tables")
    
    if not tables_dir.exists():
        print(f"Error: {tables_dir} does not exist.")
    else:
        view_tables(str(tables_dir))
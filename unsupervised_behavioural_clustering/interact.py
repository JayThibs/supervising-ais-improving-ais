from dataclasses import dataclass
from typing import List


@dataclass
class Label:
    statement: str
    cluster: int
    labels: List[str]


def add_label(statement: str, cluster: int, labels: List[str]):
    """Adds human labels to a statement."""
    label = Label(statement, cluster, labels)
    save_label(label)  # Persist to database


def search_statements(query: str) -> List[str]:
    """Search for similar statements."""
    statements = load_statements_from_db(query)
    return statements

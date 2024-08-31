import pandas as pd
from pathlib import Path
import json

RUNS_DIR = Path("data/results/runs")

def load_previous_runs(username):
    user_runs_dir = RUNS_DIR / username
    runs = {}
    if user_runs_dir.exists():
        for run_file in user_runs_dir.glob("*.json"):
            with open(run_file, "r") as f:
                runs[run_file.stem] = json.load(f)
    return runs

def save_run(username, run_name, results, config):
    user_runs_dir = RUNS_DIR / username
    user_runs_dir.mkdir(parents=True, exist_ok=True)
    run_file = user_runs_dir / f"{run_name}.json"
    
    run_data = {
        'results': results,
        'config': config
    }
    
    with open(run_file, "w") as f:
        json.dump(run_data, f)

def compare_run_results(runs):
    metrics = []
    for run in runs:
        metrics.append({
            'Run Name': run['config']['name'],
            'Num Clusters': run['results']['n_clusters'],
            'Silhouette Score': run['results']['silhouette_score'],
            'Avg Cluster Size': run['results']['avg_cluster_size'],
            'Avg Approval Score': run['results']['avg_approval_score']
        })
    
    comparison_df = pd.DataFrame(metrics)
    
    combined_results = {
        'embeddings': [],
        'labels': [],
        'run_names': []
    }
    
    for run in runs:
        combined_results['embeddings'].extend(run['results']['embeddings'])
        combined_results['labels'].extend(run['results']['labels'])
        combined_results['run_names'].extend([run['config']['name']] * len(run['results']['labels']))
    
    return {
        'metrics': comparison_df,
        'combined_results': combined_results
    }
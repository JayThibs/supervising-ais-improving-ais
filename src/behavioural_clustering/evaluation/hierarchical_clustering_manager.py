from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.evaluation.clustering import ClusterAnalyzer
from behavioural_clustering.utils.visualization import Visualization
from typing import List, Dict, Any, Optional, Tuple

class HierarchicalClusteringManager:
    def __init__(self, run_settings: RunSettings, cluster_analyzer: ClusterAnalyzer, visualization: Visualization):
        self.settings = run_settings
        self.cluster_analyzer = cluster_analyzer
        self.visualization = visualization
        self.hierarchy_data = {}

    def run_hierarchical_clustering(
        self,
        prompt_type: str,
        chosen_clustering,
        approvals_data,
        rows,
        model_names: List[str],
        approval_prompts: Dict[str, str]
    ):
        print(f"Running hierarchical clustering for {prompt_type}...")
        
        hierarchy_data = self.cluster_analyzer.calculate_hierarchical_cluster_data(
            chosen_clustering,
            approvals_data,
            rows
        )
        
        self.hierarchy_data[prompt_type] = hierarchy_data
        
        labels = list(approval_prompts.keys())
        
        if prompt_type not in self.settings.plot_settings.hide_plots:
            for model_name in model_names:
                self.visualization.visualize_hierarchical_plot(
                    hierarchy_data=hierarchy_data,
                    plot_type=prompt_type,
                    filename=f"{self.settings.directory_settings.viz_dir}/hierarchical_clustering_{prompt_type}_{model_name}",
                    labels=labels
                )
        else:
            print(f"Skipping visualization for {prompt_type} as per settings.")

        print(f"Hierarchical clustering for {prompt_type} completed.")
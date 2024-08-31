import streamlit as st
from behavioural_clustering.config.run_settings import RunSettings, ModelSettings, ClusteringSettings, PlotSettings, TsneSettings

def save_custom_configuration(config_manager, run_settings, clustering_algorithm, n_clusters, show_tsne, show_umap, 
                              show_hierarchical, show_approval, perplexity, learning_rate, n_neighbors, min_dist):
    # Update the RunSettings object with the current values
    run_settings.clustering_settings.main_clustering_algorithm = clustering_algorithm
    run_settings.clustering_settings.n_clusters = n_clusters
    run_settings.plot_settings.hide_tsne = not show_tsne
    run_settings.plot_settings.hide_hierarchical = not show_hierarchical
    run_settings.plot_settings.hide_approval = not show_approval
    run_settings.tsne_settings.perplexity = perplexity
    run_settings.tsne_settings.learning_rate = learning_rate
    
    # TODO: Add UMAP settings to RunSettings if needed
    
    # Save the configuration
    config_manager.save_configuration(run_settings)
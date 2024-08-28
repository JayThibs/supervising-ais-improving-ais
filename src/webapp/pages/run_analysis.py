import streamlit as st
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from webapp.components.sidebar import get_selected_models

def show():
    st.header("Run Analysis")
    
    run_type = st.session_state.get("run_type", "Full Evaluation")
    
    if st.button("Start Analysis"):
        config_manager = st.session_state.config_manager
        run_settings = config_manager.get_configuration(st.session_state.selected_config)
        
        # Update run settings with customized values
        run_settings.data_settings.n_statements = st.session_state.n_statements
        run_settings.model_settings.models = get_selected_models()
        run_settings.prompt_settings.max_desc_length = st.session_state.max_desc_length
        run_settings.clustering_settings.n_clusters = st.session_state.n_clusters
        run_settings.plot_settings.hide_plots = st.session_state.hide_plots
        
        evaluator = EvaluatorPipeline(run_settings)
        
        with st.spinner("Running analysis..."):
            if run_type == "Full Evaluation":
                evaluator.run_evaluations()
            elif run_type == "Model Comparison":
                metadata_config = evaluator.create_current_metadata()
                evaluator.run_model_comparison(metadata_config)
            elif run_type == "Approval Prompts":
                metadata_config = evaluator.create_current_metadata()
                for prompt_type in evaluator.approval_prompts.keys():
                    evaluator.run_prompt_evaluation(prompt_type, metadata_config)
        
        st.success("Analysis completed successfully!")
    
    # Display current settings
    st.subheader("Current Run Settings")
    st.write(f"Run Type: {run_type}")
    st.write(f"Number of Statements: {st.session_state.n_statements}")
    st.write(f"Selected Models: {', '.join([f'{m[0]}/{m[1]}' for m in get_selected_models()])}")
    st.write(f"Max Description Length: {st.session_state.max_desc_length}")
    st.write(f"Number of Clusters: {st.session_state.n_clusters}")
    st.write(f"Hidden Plots: {', '.join(st.session_state.hide_plots)}")

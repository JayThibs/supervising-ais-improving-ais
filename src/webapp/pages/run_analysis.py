import streamlit as st
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from behavioural_clustering.config.run_settings import RunSettings

def show():
    st.header("Run Analysis")

    if st.button("Run Clustering Analysis"):
        try:
            with st.spinner("Running clustering analysis..."):
                progress_bar = st.progress(0)
                
                # Get the current configuration
                run_settings = st.session_state.config_manager.get_configuration("Custom")
                
                # Initialize EvaluatorPipeline
                evaluator = EvaluatorPipeline(run_settings)
                
                # Run the evaluation pipeline
                evaluator.run_evaluations()
                
            st.success("Clustering analysis complete!")
            
            # Visualize results
            st.subheader("Visualization Results")
            
            # Display plots
            st.image("embedding_responses.png", caption="Embedding Responses")
            st.image("approval_plot.png", caption="Approval Plot")
            st.image("hierarchical_cluster.png", caption="Hierarchical Clustering")
            st.image("spectral_clustering.png", caption="Spectral Clustering")
            
            # Save run
            run_name = st.text_input("Enter a name for this run:")
            if st.button("Save Run"):
                # Implement save_run functionality using evaluator
                evaluator.save_run(run_name)
                st.success(f"Run '{run_name}' saved successfully!")
            
            # Export results
            if st.button("Export Results"):
                # Implement export_results functionality using evaluator
                evaluator.export_results()
                st.success("Results exported successfully!")
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")

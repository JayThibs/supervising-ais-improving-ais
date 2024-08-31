import streamlit as st
from behavioural_clustering.models.model_factory import initialize_model

def select_models(model_settings):
    st.subheader("Model Selection")
    
    selected_models = []
    for i, (model_family, model_name) in enumerate(model_settings.models):
        st.write(f"Model {i+1}")
        new_model_family = st.selectbox(f"Select model family for Model {i+1}", ["OpenAI", "Anthropic", "Hugging Face"], key=f"model_family_{i}")
        
        if new_model_family == "OpenAI":
            new_model = st.selectbox(f"Select OpenAI model for Model {i+1}", ["gpt-3.5-turbo", "gpt-4"], key=f"model_{i}")
        elif new_model_family == "Anthropic":
            new_model = st.selectbox(f"Select Anthropic model for Model {i+1}", ["claude-v1", "claude-instant-v1"], key=f"model_{i}")
        else:
            new_model = st.text_input(f"Enter Hugging Face model name for Model {i+1}", key=f"model_{i}")
            if st.button(f"Load Model {i+1}"):
                with st.spinner("Loading model..."):
                    try:
                        initialize_model({"model_family": "local", "model_name": new_model})
                        st.success(f"Successfully loaded {new_model}")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        
        selected_models.append((new_model_family.lower(), new_model))
    
    if st.button("Add Another Model"):
        selected_models.append(("openai", "gpt-3.5-turbo"))  # Default new model
    
    return selected_models
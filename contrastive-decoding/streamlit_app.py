import streamlit as st
import os
import sys
from dotenv import load_dotenv
import json
import time
from typing import List, Dict

# Get the directory containing the current script (streamlit_app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Running streamlit app from: {os.path.abspath(__file__)}")

# Add the current directory to the Python path
sys.path.append(current_dir)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(current_dir), '.env'))

from assistant_find_divergence_prompts import DivergenceFinder
from automated_pipeline import AutomatedPipeline
from visualizer import Visualizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import json
from ai_assistant import AIAssistant

st.set_page_config(layout="wide")
st.title("Automated Model Divergence Analysis")

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def load_prompts(topics):
    prompts = []
    prompt_topics = []
    for topic in topics:
        filename = f"assistant_prompts/{topic.lower()}_find_high_div_prompts.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                prompts.extend(data["seed_texts_1"])
                prompt_topics.extend([topic] * len(data["seed_texts_1"]))
    return prompts, prompt_topics

def save_prompts(topics, prompts):
    for topic in topics:
        filename = f"assistant_prompts/{topic.lower()}_find_high_div_prompts.json"
        if os.path.exists(filename):
            with open(filename, 'r+') as f:
                data = json.load(f)
                data["seed_texts_1"] = [prompt for prompt, prompt_topic in zip(prompts, prompt_topics) if prompt_topic == topic]
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

def run_cd_with_updates(command: List[str], output_placeholder):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    
    results = []
    for line in process.stdout:
        if line.startswith("RESULT:"):
            result = json.loads(line[7:])
            results.append(result)
            output_placeholder.json(result)
        else:
            output_placeholder.text(line.strip())
    
    return results

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Setup", "Run Automated Pipeline", "Run CD", "Run DivergenceFinder", "Results", "Visualizations", "Generate Prompts", "Manage Prompts"])

with tab1:
    st.header("Experiment Setup")
    
    st.info("Select the models you want to compare and set up the experiment parameters.")
    
    available_models = [
        "NousResearch/Meta-Llama-3-8B",
        "NousResearch/Meta-Llama-3-8B-Instruct",
        "gpt2",
        "gpt2-xl",
        "EleutherAI/gpt-j-6b",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
        "facebook/opt-66b",
        "mistralai/Mistral-7B-v0.1",
        "EleutherAI/gpt-neox-20b"
    ]
    
    model1 = st.selectbox("Select Model 1", available_models, help="The first model to use in the comparison")
    model2 = st.selectbox("Select Model 2", available_models, help="The second model to use in the comparison")
    
    topics = st.multiselect("Select Topics of Interest", ["Bias", "Factual Accuracy", "Reasoning", "Creativity"], help="Choose the topics you want to focus on in the experiment")
    num_cycles = st.slider("Number of Experiment Cycles", 1, 10, 5, help="The number of times to run the experiment")
    generation_length = st.slider("Generation Length", 10, 100, 40, help="The number of tokens to generate for each prompt")
    assistant_model = st.selectbox("Select Assistant Model", ["Local LLM", "OpenAI GPT-3.5", "OpenAI GPT-4"], help="The model to use for generating prompts and analyzing results")
    
    if assistant_model.startswith("OpenAI"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter OpenAI API Key", type="password", help="Your OpenAI API key for using GPT models")
        else:
            st.success("OpenAI API Key found in .env file")
    else:
        api_key = None

with tab2:
    st.header("Run Automated Pipeline")
    st.info("The Automated Pipeline runs a series of experiments to find divergences between the selected models.")
    
    if st.button("Run Full Automated Pipeline"):
        with st.spinner("Running Automated Pipeline..."):
            pipeline = AutomatedPipeline(model1, model2, topics, ai_model=assistant_model, api_key=api_key)
            results = pipeline.run(num_cycles=num_cycles)
        st.session_state.results = results
        st.success("Automated Pipeline completed successfully!")
        st.info("The results are now available in the Results tab.")

with tab3:
    st.header("Run Contrastive Decoding")
    
    # Model selection
    model1 = st.selectbox("Select Model 1 (Starting Model)", available_models, key="cd_model1")
    model2 = st.selectbox("Select Model 2 (Comparison Model)", available_models, key="cd_model2")
    
    # Generate default target name
    def get_model_shortname(model_name):
        return model_name.split('/')[-1].lower()
    
    default_target = f"{get_model_shortname(model1)}-{get_model_shortname(model2)}"
    
    # Target name explanation and input
    st.info("The target name is used to identify this specific CD run. It will be used in the output file name.")
    target = st.text_input("Enter target name for CD run", value=default_target, help="You can edit this name if you want", key="cd_target")
    
    # Interpolation weight explanation
    st.info("Interpolation weight mixes a fraction of the comparison model into the starting model. Values range from 0 to 1.")
    interp_weight = st.number_input("Enter interpolation weight", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="cd_interp_weight")
    
    # Prompt selection
    st.subheader("Select Prompts")
    prefix_folder = os.path.join(current_dir, "prefix_folder")
    st.write(f"Prefix folder path: {prefix_folder}")
    if not os.path.exists(prefix_folder):
        st.error(f"Prefix folder not found: {prefix_folder}")
    else:
        prefix_files = [f for f in os.listdir(prefix_folder) if f.endswith('.txt')]
        if not prefix_files:
            st.warning("No .txt files found in the prefix folder.")
        else:
            selected_prefix_file = st.selectbox("Choose a prefix file", prefix_files, key="cd_prefix_file")
            
            if selected_prefix_file:
                with open(os.path.join(prefix_folder, selected_prefix_file), 'r') as f:
                    prompts = f.read().splitlines()
                selected_prompts = st.multiselect("Select prompts to use", prompts, key="cd_selected_prompts")
    
    # Additional parameters in a collapsible section
    with st.expander("Advanced Parameters", expanded=False):
        generation_length = st.slider("Generation Length", 10, 100, 40, help="The number of tokens to generate for each prompt", key="cd_generation_length")
        batch_size = st.slider("Batch Size", 1, 64, 32, help="Number of prompts to process in parallel", key="cd_batch_size")
        starting_model_weight = st.number_input("Starting Model Weight", value=1.0, help="Weight for the starting model", key="cd_starting_model_weight")
        comparison_model_weight = st.number_input("Comparison Model Weight", value=-1.0, help="Weight for the comparison model", key="cd_comparison_model_weight")
        set_prefix_len = st.number_input("Prefix Length", value=5, min_value=1, help="Number of tokens to use as prefix", key="cd_set_prefix_len")
        divergence_fnct = st.selectbox("Divergence Function", ["l1", "kl"], help="Function to measure divergence between model outputs", key="cd_divergence_fnct")
        quantize = st.checkbox("Quantize Models", value=True, help="Use quantization to reduce memory usage", key="cd_quantize")
        cache_attn = st.checkbox("Cache Attention", value=True, help="Cache attention to speed up generation", key="cd_cache_attn")
    
    if st.button("Run Contrastive Decoding", key=f"cd_run_button_{os.path.getmtime(__file__)}"):
        st.write("Button pressed. Starting Contrastive Decoding process...")
        if not selected_prompts:
            st.warning("Please select at least one prompt.")
        else:
            st.write(f"Number of selected prompts: {len(selected_prompts)}")
            prompts_file = os.path.join(current_dir, "temp_prompts.txt")
            st.write(f"Writing prompts to temporary file: {prompts_file}")
            with open(prompts_file, 'w') as f:
                for prompt in selected_prompts:
                    f.write(prompt + "\n")
            
            # Update the path to run_CD.py
            run_cd_path = os.path.join(current_dir, "run_CD.py")
            st.write(f"Looking for run_CD.py at: {run_cd_path}")
            
            if not os.path.exists(run_cd_path):
                st.error(f"run_CD.py not found at {run_cd_path}")
            else:
                st.success(f"run_CD.py found at {run_cd_path}")
            
            command = [
                sys.executable,
                os.path.join(current_dir, "run_CD.py"),
                "--target", target,
                "--interp_weight", str(interp_weight),
                "--prefixes_path", prompts_file,
                "--model_name", model1,
                "--starting_model_path", model1,
                "--comparison_model_path", model2,
                "--generation_length", str(generation_length),
                "--batch_size", str(batch_size),
                "--starting_model_weight", str(starting_model_weight),
                "--comparison_model_weight", str(comparison_model_weight),
                "--set_prefix_len", str(set_prefix_len),
                "--divergence_fnct", divergence_fnct
            ]
            
            if quantize:
                command.append("--quantize")
            if cache_attn:
                command.append("--cache_attn")
            
            st.write("Command to be executed:", " ".join(command))
            
            output_placeholder = st.empty()
            
            with st.spinner("Running Contrastive Decoding..."):
                try:
                    results = run_cd_with_updates(command, output_placeholder)
                    st.success("Contrastive Decoding run completed!")
                    
                    # Save results to a JSON file
                    results_file = os.path.join(current_dir, f"cd_results_{int(time.time())}.json")
                    with open(results_file, 'w') as f:
                        json.dump(results, f)
                    st.success(f"Results saved to {results_file}")
                    
                    # Store results in session state for use in other tabs
                    st.session_state.cd_results = results
                    
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
            
            st.write(f"Removing temporary file: {prompts_file}")
            os.remove(prompts_file)

with tab4:
    st.header("Run DivergenceFinder")
    st.info("DivergenceFinder analyzes the output of Contrastive Decoding to identify significant divergences between models.")
    
    if 'cd_results' in st.session_state:
        st.success("Contrastive Decoding results found in memory.")
        use_memory_results = st.checkbox("Use results from memory", value=True)
    else:
        use_memory_results = False
    
    if not use_memory_results:
        cd_output_file = st.text_input("Enter path to CD output file", help="The file containing the output from the Contrastive Decoding run")
    
    if st.button("Run DivergenceFinder"):
        if use_memory_results:
            cd_output = st.session_state.cd_results
        elif not os.path.exists(cd_output_file):
            st.error(f"File not found: {cd_output_file}")
        else:
            with open(cd_output_file, 'r') as f:
                cd_output = json.load(f)
        
        if cd_output:
            divergence_finder = DivergenceFinder(
                model_name=model1,
                starting_model_path=model1,
                comparison_model_path=model2,
                generation_length=generation_length,
                n_cycles_ask_assistant=num_cycles,
                ai_model=assistant_model if assistant_model.startswith("OpenAI") else model1
            )
            
            with st.spinner("Running DivergenceFinder..."):
                results = divergence_finder.search_loop(cd_output)
            
            st.session_state.results = results
            st.success("DivergenceFinder completed successfully!")
            st.info("The results are now available in the Results tab.")

with tab5:
    st.header("Results")
    if 'results' in st.session_state:
        results = st.session_state.results
        if isinstance(results, list):  # DivergenceFinder results
            st.info("These results show the divergences found between the models.")
            sort_by = st.selectbox("Sort results by", ["Divergence", "Custom Score", "Cycle"])
            ascending = st.checkbox("Sort ascending")
            
            sorted_results = sorted(results[0], key=lambda x: x[["Divergence", "Custom Score", "Cycle"].index(sort_by)], reverse=not ascending)
            
            for i, (divergence, text, response, custom_score, cycle, _) in enumerate(sorted_results):
                with st.expander(f"Result {i+1} - Divergence: {divergence:.4f}, Custom Score: {custom_score:.4f}, Cycle: {cycle}"):
                    st.write(f"**Text:** {text}")
                    st.write(f"**Response:** {response}")
        else:  # AutomatedPipeline results
            st.write(results)  # Assuming the report is a string or can be displayed directly
    else:
        st.info("Run an experiment to see results here.")

with tab6:
    st.header("Visualizations")
    if 'results' in st.session_state:
        results = st.session_state.results
        if isinstance(results, list):  # DivergenceFinder results
            st.info("These visualizations show trends in the divergences found between the models.")
            divergences = [r[0] for r in results[0]]
            cycles = [r[4] for r in results[0]]
            custom_scores = [r[3] for r in results[0]]
            
            chart_type = st.selectbox("Select chart type", ["Divergence Trends", "Custom Score vs Divergence", "Cycle vs Divergence"])
            
            if chart_type == "Divergence Trends":
                fig = Visualizer.plot_divergence_trends(divergences, cycles)
            elif chart_type == "Custom Score vs Divergence":
                fig = Visualizer.plot_custom_score_vs_divergence(custom_scores, divergences)
            else:
                fig = Visualizer.plot_cycle_vs_divergence(cycles, divergences)
            
            st.plotly_chart(fig)
        else:  # AutomatedPipeline results
            st.info("Visualizations for AutomatedPipeline not implemented yet.")
    else:
        st.info("Run an experiment to see visualizations here.")

with tab7:
    st.header("Generate New Prompts")
    st.info("This tab allows you to generate new prompts using the selected assistant model.")
    
    if st.button("Generate Prompts"):
        ai_assistant = AIAssistant(model="gpt-4")
        new_prompts = []
        
        for topic in topics:
            prompt = f"Generate 3 prompts that could lead to high divergence between language models on the topic of {topic}. Each prompt should be a single sentence or question."
            response = ai_assistant.generate(prompt, max_tokens=200)
            if response:
                new_prompts.extend(response.split('\n'))
        
        st.subheader("Generated Prompts:")
        for i, prompt in enumerate(new_prompts, 1):
            st.write(f"{i}. {prompt}")
        
        if st.button("Add to Existing Prompts"):
            prefix_folder = os.path.join(current_dir, "prefix_folder")
            if not os.path.exists(prefix_folder):
                os.makedirs(prefix_folder)
            
            filename = os.path.join(prefix_folder, "generated_prompts.txt")
            with open(filename, 'a') as f:
                for prompt in new_prompts:
                    f.write(prompt + "\n")
            
            st.success("New prompts added and saved successfully!")

with tab8:
    st.header("Manage Prompts")
    st.info("This tab allows you to view, edit, and manage existing prompts.")
    
    prefix_folder = os.path.join(current_dir, "prefix_folder")
    if not os.path.exists(prefix_folder):
        st.error(f"Prefix folder not found: {prefix_folder}")
    else:
        prefix_files = [f for f in os.listdir(prefix_folder) if f.endswith('.txt')]
        if not prefix_files:
            st.warning("No .txt files found in the prefix folder.")
        else:
            selected_prefix_file = st.selectbox("Choose a prefix file", prefix_files)
            
            if selected_prefix_file:
                file_path = os.path.join(prefix_folder, selected_prefix_file)
                with open(file_path, 'r') as f:
                    existing_prompts = f.read().splitlines()
                
                # Display existing prompts
                st.subheader("Existing Prompts")
                
                # Add search functionality
                search_term = st.text_input("Search prompts", "")
                filtered_prompts = [p for p in existing_prompts if search_term.lower() in p.lower()]
                
                # Pagination
                prompts_per_page = 10
                page_number = st.number_input("Page", min_value=1, max_value=(len(filtered_prompts) - 1) // prompts_per_page + 1, value=1)
                start_idx = (page_number - 1) * prompts_per_page
                end_idx = start_idx + prompts_per_page
                
                # Display prompts for the current page
                for i, prompt in enumerate(filtered_prompts[start_idx:end_idx], start=start_idx):
                    st.text(f"{i+1}. {prompt}")
                
                # Edit prompt
                edit_idx = st.number_input("Enter the number of the prompt you want to edit", min_value=1, max_value=len(filtered_prompts), value=1)
                edit_prompt = st.text_input("Edit prompt", value=filtered_prompts[edit_idx-1])
                if st.button("Save Edit"):
                    filtered_prompts[edit_idx-1] = edit_prompt
                    existing_prompts[existing_prompts.index(filtered_prompts[edit_idx-1])] = edit_prompt
                    st.success("Prompt edited successfully!")
                
                # Delete prompt
                delete_idx = st.number_input("Enter the number of the prompt you want to delete", min_value=1, max_value=len(filtered_prompts), value=1)
                if st.button("Delete Prompt"):
                    deleted_prompt = filtered_prompts.pop(delete_idx-1)
                    existing_prompts.remove(deleted_prompt)
                    st.success("Prompt deleted successfully!")
                
                # Add new prompt
                st.subheader("Add New Prompt")
                new_prompt = st.text_input("Enter a new prompt")
                if st.button("Add Prompt"):
                    if new_prompt:
                        existing_prompts.append(new_prompt)
                        st.success("New prompt added successfully!")
                    else:
                        st.warning("Please enter a prompt before adding.")
                
                # Save changes
                if st.button("Save All Changes"):
                    with open(file_path, 'w') as f:
                        for prompt in existing_prompts:
                            f.write(prompt + "\n")
                    st.success("All changes saved successfully!")
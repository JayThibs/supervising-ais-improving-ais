import os
import json
import numpy as np
import pickle
from datetime import datetime
import pdb
from matplotlib import pyplot as plt
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization
from clustering import Clustering
from utils import embed_texts, load_pkl_or_not, query_model_on_statements
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ModelMetadata:
    model_id: str
    temperature: float
    max_tokens: int
    other_params: Dict[str, Any]  # Include any other model-specific parameters


@dataclass
class StatementResponses:
    metadata: ModelMetadata
    response_file: str  # Path to the .jsonl file with responses

    def save_responses(self, statements, responses):
        with open(self.response_file, "w") as f:
            for statement, response in zip(statements, responses):
                json.dump({"statement": statement, "response": response}, f)
                f.write("\n")

    @staticmethod
    def load_responses(filepath):
        statements, responses = [], []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                statements.append(data["statement"])
                responses.append(data["response"])
        return statements, responses


class EvaluatorPipeline:
    def __init__(self, args):
        self.args = args
        # Set up directories
        self.test_mode = args.test_mode
        self.data_dir = f"{os.getcwd()}/data"
        self.evals_dir = f"{self.data_dir}/evals"
        self.results_dir = f"{self.data_dir}/results"
        self.pickle_dir = f"{self.results_dir}/pickle_files"
        self.viz_dir = f"{self.results_dir}/plots"
        self.tables_dir = f"{self.results_dir}/tables"

        # model information
        self.llms = []
        print(f"self.args.model: {self.args.model}")
        print(f"self.args.model type: {type(self.args.model)}")
        for model in self.args.model:
            if "gpt-" in model and (
                not any(char.isdigit() for char in model.split("-")[-1][0])
                or "turbo" in model
            ):
                model_family = "openai"
            elif "claude" in model:
                model_family = "anthropic"
            else:
                model_family = "local"
            self.llms.append((model_family, model))

        # Load approval prompts from same directory as this file
        self.n_statements = 300 if self.test_mode else self.args.n_statements
        with open(f"{self.data_dir}/prompts/approval_prompts.json", "r") as file:
            # { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...}
            self.approval_prompts = json.load(file)
            self.persona_approval_prompts = [
                prompt for prompt in list(self.approval_prompts["personas"].values())
            ]
        # Create a loop that creates a new attribute which is a dictionary of the
        # persona prompt title and the prompt itself
        for key in self.approval_prompts.keys():
            setattr(self, key, list(self.approval_prompts[key].keys()))

        # with open(f"{os.getcwd()}/data/prompts/approval_prompts.json", "r") as file:
        #     self.approval_prompts = json.load(file)
        #     for key in self.approval_prompts.keys():
        #         # This creates labels for personas, awareness, and whatever else is in the json file.
        #         # For example, if the json file has a key "awareness", this will create a self.awareness
        #         # attribute with the values of the "awareness" key in the json file.
        #         setattr(self, key, list(self.approval_prompts[key].keys()))
        self.approval_question_prompt_template = self.args.approval_prompt_template
        self.set_reuse_flags()
        self.hide_plots = self.process_hide_plots(self.args.hide_plots)
        self.plot_statement_clustering = False

        # Set up objects
        self.model_eval = ModelEvaluation(args, self.llms)
        self.viz = Visualization(save_path=self.viz_dir)
        self.clustering_obj = Clustering(self.args)

        if self.args.new_generation:
            self.saved_query_results = None
            if "all_query_results.pkl" in os.listdir(self.pickle_dir):
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                os.rename(
                    f"{self.pickle_dir}/all_query_results.pkl",
                    f"{self.pickle_dir}/all_query_results_{timestamp}.pkl",
                )
        else:
            if "all_query_results.pkl" in os.listdir(self.pickle_dir):
                self.saved_query_results = self.load_results(
                    "all_query_results.pkl", "pickle_files"
                )
            else:
                self.query_results = None

    def setup_evaluations(self):
        self.setup_directories()
        self.load_api_key()
        self.clone_evals_repo()

    # Set up directories
    def setup_directories(self):
        dirs = [
            self.data_dir,
            self.evals_dir,
            self.results_dir,
            self.viz_dir,
            self.tables_dir,
            self.pickle_dir,
        ]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_api_key(self):
        self.api_key = DataPreparation.load_api_key(self, "OPENAI_API_KEY")

    def clone_evals_repo(self):
        if not os.path.exists(
            os.path.join(self.evals_dir, "anthropic-model-written-evals")
        ):
            DataPreparation.clone_repo(
                self,
                "https://github.com/anthropics/evals.git",
                "anthropic-model-written-evals",
            )

    def load_evaluation_data(self, dataset_name="anthropic"):
        self.dataset_name = dataset_name
        self.data_prep = DataPreparation()
        all_texts = self.data_prep.load_evaluation_data(self.data_prep.file_paths)
        return all_texts

    def process_reuse_data(self, reuse_data, all_data_types):
        reuse_types = set()

        if "all" in reuse_data:
            reuse_types = set(all_data_types)  # Ensure it's a set
            for item in reuse_data:
                if item.startswith("!"):
                    exclude_type = item[1:]
                    reuse_types.discard(exclude_type)  # discard method works on sets
        else:
            for item in reuse_data:
                if item in all_data_types:
                    reuse_types.add(item)  # add method to include in the set

        return reuse_types

    def set_reuse_flags(self):
        data_types = [
            "embedding_clustering",
            "joint_embeddings",
            "tsne",
            "approvals",
            "hierarchical_approvals",
            "hierarchical_awareness",
            "awareness",
            "cluster_rows",
            "conditions",
        ]

        reuse_data_types = self.process_reuse_data(self.args.reuse_data, data_types)
        print("Data types to reuse:", reuse_data_types)

        for data_type in data_types:
            setattr(self, f"reuse_{data_type}", data_type in reuse_data_types)

    def process_hide_plots(self, hide_plots):
        all_plot_types = [
            "tsne",
            "approval",
            "awareness",
            "hierarchical_approvals",
            "hierarchical_awareness",
            "spectral",
        ]
        hide_types = []

        if "all" not in hide_plots:
            hide_types = all_plot_types
        else:
            for plot_type in hide_plots:
                if plot_type.startswith("!"):
                    include_plot_type = plot_type[1:]
                    if include_plot_type in all_plot_types:
                        hide_types.remove(include_plot_type)

        return hide_types

    def save_results(self, data, file_name, sub_dir):
        # Save data to a pickle file
        if not os.path.exists(f"{self.results_dir}/{sub_dir}"):
            os.makedirs(f"{self.results_dir}/{sub_dir}")
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "wb") as f:
            pickle.dump(data, f)

    def load_results(self, file_name, sub_dir):
        # Load data from a pickle file
        try:
            with open(f"{self.results_dir}/{sub_dir}/{file_name}", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def generate_plot_filename(self, model_names: list, plot_type: str):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        plot_type = plot_type.replace(" ", "_")
        filename = f"{self.viz_dir}/"
        for model in model_names:
            filename += f"{model}-"
        filename += f"{plot_type}.png"
        return filename

    def generate_responses(self):
        self.perplexity = 3 if self.test_mode else 50
        text_subset = self.data_prep.load_and_preprocess_data(
            n_statements=self.n_statements
        )

        if self.saved_query_results is None:
            all_query_results = self.generate_and_save_responses(
                text_subset,
                self.n_statements,
                self.args.statements_prompt_template,
                self.args.statements_system_message,
                self.llms,
            )
        else:
            all_query_results = [self.saved_query_results]

        return text_subset, all_query_results

    def generate_and_save_responses(
        self, text_subset, n_statements, prompt_template, system_message, llms
    ):
        all_query_results = []
        for model_family, model in llms:
            print(f"Generating responses for {model} from {model_family}...")
            file_name = f"{model_family}_{model}_reaction_to_{n_statements}_{self.dataset_name}_statements.pkl"
            query_results = query_model_on_statements(
                text_subset, model_family, model, prompt_template, system_message
            )  # dictionary of inputs, responses, and model instance
            all_query_results.append(query_results)
            self.save_results(query_results, file_name, "pickle_files")
            print(f"{file_name} saved.")
        self.save_results(
            all_query_results, "all_query_results.pickle", "pickle_files"
        )  # last saved
        print(f"{file_name} saved.")
        return all_query_results

    def collect_model_info(self, all_query_results):
        print("Collecting model info...")
        print(f"all_query_results: {all_query_results}")
        all_model_info = [result["model_info"] for result in all_query_results]
        # print the keys
        for key in all_model_info[0].keys():
            print(key)
        return all_model_info

    def perform_tsne_dimensionality_reduction(self, combined_embeddings):
        print("Performing t-SNE dimensionality reduction...")
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            combined_embeddings, iterations=300, perplexity=self.perplexity
        )
        self.check_tsne_values(dim_reduce_tsne)
        return dim_reduce_tsne

    def check_tsne_values(self, dim_reduce_tsne):
        if not np.isfinite(dim_reduce_tsne).all():
            print("dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("dim_reduce_tsne contains NaN or inf values.")
        print("dim_reduce_tsne:", dim_reduce_tsne.dtype)

    def visualize_results(
        self, dim_reduce_tsne, joint_embeddings_all_llms, model_names, tsne_filename
    ):
        if "tsne" not in self.hide_plots:
            self.viz.plot_embedding_responses(
                dim_reduce_tsne,
                joint_embeddings_all_llms,
                model_names,
                tsne_filename,
            )

    def embed_responses(self, all_query_results):
        print("Embedding responses...")
        file_loaded, joint_embeddings_all_llms = load_pkl_or_not(
            "joint_embeddings_all_llms.pkl",
            self.pickle_dir,
            self.reuse_joint_embeddings,
        )
        if not file_loaded:
            joint_embeddings_all_llms = self.create_embeddings(
                all_query_results, self.llms
            )

        combined_embeddings = np.array(
            [e[3] for e in joint_embeddings_all_llms]
        )  # grab the embeddings of the inputs + responses
        combined_embeddings = np.array(combined_embeddings, dtype=np.float64)
        if not np.isfinite(combined_embeddings).all():
            print("Embeddings contain non-finite values.")

        return joint_embeddings_all_llms, combined_embeddings

    def create_embeddings(
        self,
        all_query_results,
        llms,
        embedding_model="text-embedding-ada-002",
        combine_statements=False,
        save=True,
    ):
        """Embed the responses generated by ."""
        joint_embeddings_all_llms = []

        for i, (model_family, model) in enumerate(llms):
            print(f"Embedding responses for LLM {i}...")
            inputs = all_query_results[i]["inputs"]  # list of statements
            responses = all_query_results[i][
                "responses"
            ]  # list of responses to the statements by the LLM number i
            print(f"inputs: {inputs}")
            print(f"responses: {responses}")
            if i == 0:
                inputs_embeddings = embed_texts(texts=inputs, model=embedding_model)
                n_statements = len(inputs)
                with open(
                    f"{self.pickle_dir}/{self.dataset_name}_{n_statements}_statements_embs.pkl",
                    "wb",
                ) as f:
                    pickle.dump(inputs_embeddings, f)

            # Embed the model responses to the input statements of a given model
            if combine_statements:
                joint_embeddings = embed_texts(
                    texts=[
                        input + " " + response
                        for input, response in zip(inputs, responses)
                    ],
                    model=embedding_model,
                )
            else:
                responses_embeddings = embed_texts(
                    texts=responses, model=embedding_model
                )
                joint_embeddings = [
                    i + r for i, r in zip(inputs_embeddings, responses_embeddings)
                ]
            for input, response, embedding in zip(inputs, responses, joint_embeddings):
                joint_embeddings_all_llms.append([i, input, response, embedding, model])

        if save:
            with open(f"{self.pickle_dir}/joint_embeddings_all_llms.pkl", "wb") as f:
                pickle.dump(joint_embeddings_all_llms, f)
        return joint_embeddings_all_llms

    def run_clustering(self, combined_embeddings):
        file_loaded, chosen_clustering = load_pkl_or_not(
            "chosen_clustering.pkl", self.pickle_dir, self.reuse_embedding_clustering
        )
        if not file_loaded:
            print("Running clustering...")
            self.clustering_results = self.clustering_obj.cluster_embeddings(
                combined_embeddings, multiple=True
            )
            print("Choosing clustering method... (KMeans is default)")
            chosen_clustering = self.clustering_results["KMeans"]
            with open(f"{self.pickle_dir}/chosen_clustering.pkl", "wb") as f:
                pickle.dump(chosen_clustering, f)
        return chosen_clustering

    def analyze_response_embeddings_clusters(
        self, chosen_clustering, joint_embeddings_all_llms, all_query_results
    ):
        print("Analyzing clusters...")
        file_loaded, rows = load_pkl_or_not(
            "rows.pkl", self.pickle_dir, self.reuse_cluster_rows
        )
        if not file_loaded:
            rows = self.generate_and_save_cluster_analysis(
                chosen_clustering, joint_embeddings_all_llms, all_query_results
            )
        # Save and display the results
        print("Saving and displaying results...")
        self.model_eval.save_and_display_results(
            chosen_clustering, rows, self.all_model_info
        )
        return rows

    def load_or_generate_approvals_data(
        self,
        approvals_type,
        statement_embeddings,
        reuse_approvals=False,
    ):
        """
        Using the statements from a dataset, prompt the language model with a set of personas
        and ask them if they approve or disapprove of the statements. For the set of personas,
        generate the approval responses for each language model you want to evaluate.
        """
        pickle_filename = f"approvals_statements_and_embeddings_{approvals_type}.pkl"
        file_loaded = False
        if reuse_approvals:
            file_loaded, approvals_statements_and_embeddings = load_pkl_or_not(
                pickle_filename,
                self.pickle_dir,
                reuse_approvals,
            )

        if not file_loaded:
            print("Generating approvals...")

            # load statement embeddings
            with open(
                f"{self.pickle_dir}/{self.dataset_name}_{self.n_statements}_statements_embs.pkl",
                "rb",
            ) as f:
                statement_embeddings = pickle.load(f)

            conditions_loaded, all_condition_approvals = load_pkl_or_not(
                "all_condition_approvals_{approvals_type}.pkl",
                self.pickle_dir,
                self.reuse_conditions,
            )

            if not self.reuse_conditions or not conditions_loaded:
                # note: self.args.model is a list of models
                # create a dictionary with the model as the key, and a list of persona approvals as the value
                all_condition_approvals = {model: [] for model in self.args.model}

                for model_family, model in self.llms:
                    model_approvals = []
                    for role_description in self.persona_approval_prompts:
                        approvals_for_statements = self.model_eval.get_model_approvals(
                            statements=self.text_subset,
                            prompt_template=self.approval_question_prompt_template,
                            model_family=model_family,
                            model=model,
                            system_message=role_description,
                        )
                        model_approvals.append(approvals_for_statements)
                    all_condition_approvals[model] = (
                        model_approvals  # {model_1: [approval1, approval2, ...], model_2: [approval1, approval2, ...], ... }
                    )

                if not os.path.exists(f"{self.pickle_dir}/all_condition_approvals.pkl"):
                    pickle.dump(
                        all_condition_approvals,
                        open(
                            f"{self.pickle_dir}/all_condition_approvals.pkl",
                            "wb",
                        ),
                    )
            print(f"all_condition_approvals: {all_condition_approvals}")

            approvals_statements_and_embeddings = []
            for i in range(len(self.text_subset)):
                print(f"Approvals record {i}")
                record_approvals = {}
                for model, model_approvals in all_condition_approvals.items():
                    record_approvals[model] = [
                        approval[i] for approval in model_approvals
                    ]
                print(f"record_approvals: {record_approvals}")
                record = [
                    record_approvals,
                    self.text_subset[i],
                    statement_embeddings[i],
                ]
                print(f"record: {record}")
                approvals_statements_and_embeddings.append(record)
                # [ [{model_1: [approval1, approval2, ...], model_2: [approval1, approval2, ...], ...}, statement, embedding], ... ]
                pickle.dump(
                    approvals_statements_and_embeddings,
                    open(
                        f"{self.pickle_dir}/{pickle_filename}",
                        "wb",
                    ),
                )

        return approvals_statements_and_embeddings

    def generate_and_save_cluster_analysis(
        self, chosen_clustering, joint_embeddings_all_llms, all_query_results
    ):
        rows = self.model_eval.analyze_clusters(
            chosen_clustering,
            joint_embeddings_all_llms,
            all_query_results,
        )
        with open(f"{self.pickle_dir}/rows.pkl", "wb") as f:
            pickle.dump(rows, f)
        return rows

    def perform_hierarchical_clustering(
        self,
        statement_clustering,
        approvals_statements_and_embeddings,
        rows,
        prompt_type,
        reuse_hierarchical_approvals,
    ):
        print("Calculating hierarchical cluster data...")
        file_loaded, hierarchy_data = load_pkl_or_not(
            f"hierarchy_approval_data_{prompt_type}.pkl",
            self.pickle_dir,
            reuse_hierarchical_approvals,
        )
        if not file_loaded:
            hierarchy_data = self.clustering_obj.calculate_hierarchical_cluster_data(
                statement_clustering,
                approvals_statements_and_embeddings,
                rows,
            )
            with open(
                f"{self.pickle_dir}/hierarchy_approval_data_{prompt_type}.pkl", "wb"
            ) as f:
                pickle.dump(hierarchy_data, f)

        return hierarchy_data

    def visualize_hierarchical_clusters(
        self, *, model_names, hierarchy_data, plot_type, labels
    ):
        """
        Visualizes hierarchical clusters for the given plot type.

        Parameters:
        - model_names: List of model names to include in the visualization.
        - hierarchy_data: The hierarchical data to be visualized.
        - plot_type: The type of plot to generate ('approval' or 'awareness').
        """
        if plot_type not in self.hide_plots:
            for model_name in model_names:
                filename = (
                    f"{self.viz_dir}/hierarchical_clustering_{plot_type}_{model_name}"
                )
                print(f"Visualizing hierarchical cluster for {model_name}...")
                self.viz.visualize_hierarchical_cluster(
                    hierarchy_data,
                    plot_type=plot_type,
                    filename=filename,
                    labels=labels,
                    bar_height=0.7,
                    bb_width=40,
                    x_leftshift=0,
                    y_downshift=0,
                )

    def load_conditions_and_embeddings(
        self,
        emb_pkl_file="Anthropic_5000_random_statements_embs",
    ):
        with open(f"{self.pickle_dir}/conditions.pkl", "rb") as f:
            be_nice_conditions = pickle.load(f)  # [ [1, 1, 1, -1 ], [0, 0, 0, 0], ... ]

        with open(f"{self.pickle_dir}/{emb_pkl_file}.pkl", "rb") as f:
            random_statements_embs = pickle.load(f)[
                : self.n_statements
            ]  # [ [statement, jsonl_filepath, [embedding]], ...]

        return be_nice_conditions, random_statements_embs

    def prepare_data_for_prompts(self, be_nice_conditions, random_statements_embs):
        approval_data_with_personas = [[[], s[0], s[2]] for s in random_statements_embs]

        for i, condition in enumerate(be_nice_conditions):
            for record in condition:
                if record == 0:
                    approval_data_with_personas[i][0].append(0)
                elif record == 1:
                    approval_data_with_personas[i][0].append(1)
                else:
                    approval_data_with_personas[i][0].append(-1)

        statement_embeddings = np.array([e[2] for e in approval_data_with_personas])

        return approval_data_with_personas, statement_embeddings

    def visualize_approval_embeddings(
        self,
        model_names,
        dim_reduce_tsne,
        approval_data,
        prompt_approver_type,  # "personas" or "awareness"
    ):
        for model_name in model_names:
            for condition in [1, 0, -1]:
                condition_title = {
                    1: "approvals",
                    0: "disapprovals",
                    -1: "no response",
                }[condition]
                plot_type = prompt_approver_type + "-" + condition_title
                approval_filename = self.generate_plot_filename(
                    model_names=[model_name], plot_type=plot_type
                )
                self.viz.plot_approvals(
                    dim_reduce_tsne,
                    approval_data,
                    condition,
                    plot_type=(
                        "approval"
                        if prompt_approver_type == "personas"
                        else "awareness"
                    ),
                    filename=f"{approval_filename}",
                    title=f"Embeddings of {condition_title} for {prompt_approver_type} responses",
                )

    def run_evaluations(self):
        """Steps to run the evaluation pipeline.

        1. Compare how multiple LLMs fall into different clusters based on their responses to the same statement prompts.
        1.1. Load statement prompts to generate model responses.
        1.2. Generate responses to statement prompts.
        1.3. Embed model responses to statement prompts.
        1.4. Run clustering on the statement + response embeddings and visualize the clusters (with spectral clustering).
        1.5. Apply dimensionality reduction to the embeddings and visualize the results.
        1.6. Analyze the clusters by auto-labeling clusters with an LLM and print and save the cluster table results.

        2. Evaluation based on prompt types (e.g., personas, awareness): How do LLMs respond to different types of prompts?
        2.1. Load prompts (approval or awareness) and embeddings for each prompt type.
        2.2. Ask the LLMs if they approve or disapprove of certain statements based on the prompt type.
        2.3. Store the LLMs' approval or disapproval responses along with the statement embeddings.
        2.4. Utilize dimensionality reduction on the statement embeddings to visualize the responses (approve, disapprove, no response) for each prompt type.
        2.5. Conduct a comparison analysis between the different prompt types.
        2.6. Generate a table to compare the approval rates of the LLMs for each cluster, segmented by prompt type.
        2.7. Perform hierarchical clustering on the responses to each prompt type and visualize the resulting clusters.
        """
        # Load data
        all_texts = self.load_evaluation_data()
        # Generate responses to statement prompts
        self.text_subset, all_query_results = self.generate_responses()
        self.all_model_info = self.collect_model_info(all_query_results)
        model_names = [llm[1] for llm in self.llms]
        tsne_filename = self.generate_plot_filename(
            model_names, "tsne_embedding_responses"
        )
        # Embed model responses to statement prompts
        joint_embeddings_all_llms, combined_embeddings = self.embed_responses(
            all_query_results
        )
        dim_reduce_tsne = self.perform_tsne_dimensionality_reduction(
            combined_embeddings
        )
        self.visualize_results(
            dim_reduce_tsne, joint_embeddings_all_llms, model_names, tsne_filename
        )
        chosen_clustering = self.run_clustering(combined_embeddings)

        labels = chosen_clustering.labels_
        print(f"labels: {labels}")
        plt.hist(labels, bins=5)
        # plt.show()
        # plt.close()

        rows = self.analyze_response_embeddings_clusters(
            chosen_clustering, joint_embeddings_all_llms, all_query_results
        )
        clusters_desc_table = [
            ["ID", "N", "Inputs Themes", "Responses Themes", "Interaction Themes"]
        ]
        table_pickle_path = f"{self.pickle_dir}/clusters_desc_table_personas.pkl"
        self.clustering_obj.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, "response_comparisons"
        )

        # ### Approval Persona prompts ###
        # approval_data_for_persona_prompts = self.load_or_generate_approvals_data(
        #     approvals_type="personas",
        #     statement_embeddings_filename=f"{self.dataset_name}_{self.n_statements}_statements_embs",
        #     reuse_approvals=self.reuse_approvals,
        # )
        # (
        #     statement_embeddings,
        #     dim_reduce_tsne,
        # ) = self.run_approvals_based_evaluation(approval_data_for_persona_prompts)
        # print("Performing dimensionality reduction...")
        # statement_embeddings = np.array([approval[2] for approval in approval_data])
        # np.save(
        #     f"{os.getcwd()}/data/results/statement_embeddings.npy",
        #     statement_embeddings,
        # )
        # dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
        #     statement_embeddings, iterations=2000, perplexity=self.perplexity
        # )
        # conditions = np.array([approval[0] for approval in approval_data])

        # pickle.dump(
        #     conditions,
        #     open(f"{os.getcwd()}/data/results/pickle_files/conditions.pkl", "wb"),
        # )
        # print(f"approval_data_persona_prompts: {approval_data_for_persona_prompts}")
        # # Visualize approval embeddings
        # if "approvals" not in self.hide_plots:
        #     self.visualize_approval_embeddings(
        #         model_names,
        #         dim_reduce_tsne,
        #         approval_data_for_persona_prompts,
        #         prompt_approver_type="personas",
        #     )
        # statement_clustering = self.run_approvals_clustering(
        #     statement_embeddings, approval_data_for_persona_prompts
        # )
        # hierarchy_data = self.perform_hierarchical_clustering(
        #     statement_clustering, approval_data_for_persona_prompts, rows
        # )
        # self.visualize_hierarchical_clusters(
        #     model_names=model_names,
        #     hierarchy_data=hierarchy_data,
        #     plot_type="approval",
        # )

        ### Awareness prompts ###
        # Load conditions and embeddings
        (
            be_nice_conditions,
            random_statements_embs,
        ) = self.load_conditions_and_embeddings()
        # Prepare data for prompts
        (
            approval_data_awareness_prompts,
            statement_embeddings,
        ) = self.prepare_data_for_prompts(be_nice_conditions, random_statements_embs)
        # Cluster statement embeddings
        statement_clustering = self.clustering_obj.cluster_statement_embeddings(
            statement_embeddings,
            prompt_approver_type="Awareness",
            spectral_plot=False if "spectral" in self.hide_plots else True,
        )
        # # Visualize awareness embeddings
        # if "awareness" not in self.hide_plots:
        #     self.visualize_approval_embeddings(
        #         model_names,
        #         dim_reduce_tsne,
        #         approval_data_awareness_prompts,
        #         prompt_approver_type="awareness",
        #     )
        # Create or load the cluster rows
        # Calculate and save hierarchical data
        # print("Calculating hierarchical cluster data for awareness prompts...")
        # hierarchy_data = self.perform_hierarchical_clustering(
        #     statement_clustering,
        #     approval_data_awareness_prompts,
        #     rows,
        # )
        # print("Visualizing hierarchical cluster...")
        # self.visualize_hierarchical_clusters(
        #     model_names=model_names,
        #     hierarchy_data=hierarchy_data,
        #     plot_type="awareness",
        # )

        # The Approval Persona prompts and Awareness prompts sections are similar, so we can refactor them into a single function where we loop over the type of prompts created in the json file. So, the following code should be able to run n number of prompts for m number of models and p number of prompt types (e.g. personas, awareness, etc.).
        # In order to do this, we will need to refactor some of the functions used for both prompt types to be more general. We will also need to allow for iterating over the models we want to evaluate.

        for prompt_type in self.approval_prompts.keys():
            print(f"prompt_type: {prompt_type}")  # e.g. "personas", "awareness", etc.
            # approvals_statements_and_embeddings: # [ [{model_1: [approval1, approval2, ...], model_2: [approval1, approval2, ...], ...}, statement, embedding], ... ]
            approvals_statements_and_embeddings = self.load_or_generate_approvals_data(
                approvals_type=prompt_type,
                statement_embeddings_pkl_filename=f"{self.dataset_name}_{self.n_statements}_statements_embs",
                reuse_approvals=self.reuse_approvals,
            )
            if "statement_embeddings" not in locals():
                statement_embeddings = np.array(
                    [approval[2] for approval in approvals_statements_and_embeddings]
                )

            if "dim_reduce_tsne" not in locals():
                print("Performing dimensionality reduction...")
                dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
                    statement_embeddings, iterations=2000, perplexity=self.perplexity
                )

            # Visualize approval embeddings
            if prompt_type not in self.hide_plots or "all" in self.hide_plots:
                self.visualize_approval_embeddings(
                    model_names,
                    dim_reduce_tsne,
                    approvals_statements_and_embeddings,
                    prompt_approver_type=prompt_type,
                )
            prompt_dict = {
                k: v for k, v in self.approval_prompts.items() if k == prompt_type
            }  # { "personas": { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...} }
            prompt_labels = list(prompt_dict.keys())

            print(f"Clustering statement embeddings...")
            n_clusters = 120
            if "statement_clustering" not in locals():
                statement_clustering = self.clustering_obj.cluster_embeddings(
                    statement_embeddings,
                    cluster_type="SpectralClustering",
                    n_clusters=n_clusters,
                    multiple=False,
                )
            if "spectral" not in self.hide_plots:
                self.viz.plot_spectral_clustering(
                    statement_clustering.labels_,
                    n_clusters=n_clusters,
                    prompt_approver_type=prompt_type.capitalize(),
                )
            self.clustering_obj.cluster_approval_stats(
                approvals_statements_and_embeddings,
                statement_clustering,
                self.all_model_info,
                prompt_dict=prompt_dict,
                reuse_cluster_rows=self.reuse_cluster_rows,
            )
            print(f"Calculating hierarchical cluster data for {prompt_type} prompts...")
            hierarchy_data = self.perform_hierarchical_clustering(
                statement_clustering,
                approvals_statements_and_embeddings,
                rows,
                prompt_type,
                reuse_hierarchical_approvals=self.reuse_hierarchical_approvals,
            )
            print(f"Visualizing hierarchical cluster for {prompt_type} prompts...")
            self.visualize_hierarchical_clusters(
                model_names=model_names,
                hierarchy_data=hierarchy_data,
                plot_type=prompt_type,
                labels=prompt_labels,
            )

        print("Done. Please check the results directory for the plots.")

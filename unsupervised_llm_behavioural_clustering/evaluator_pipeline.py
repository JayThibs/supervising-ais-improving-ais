import argparse
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization


class EvaluatorPipeline:
    def __init__(self, args):
        self.data_prep = DataPreparation()
        self.model_eval = ModelEvaluation()
        self.viz = Visualization()
        self.args = args

    def setup(self):
        self.api_key = self.data_prep.load_api_key("OPENAI_API_KEY")
        self.data_prep.clone_repo(
            "https://github.com/anthropics/evals.git", "data/evals"
        )
        self.all_texts = self.data_prep.load_evaluation_data(self.args.file_paths)

    def run_evaluation(self):
        generation_results = self.model_eval.generate_responses(
            self.all_texts[: self.args.texts_subset], self.args.llm, self.args.prompt
        )
        joint_embeddings_all_llms = self.model_eval.embed_responses(generation_results)
        clustering = self.model_eval.perform_clustering(joint_embeddings_all_llms)
        rows = self.model_eval.analyze_clusters(joint_embeddings_all_llms, clustering)
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            joint_embeddings_all_llms, iterations=2000, p=50
        )
        self.viz.plot_dimension_reduction(dim_reduce_tsne)
        self.viz.visualize_hierarchical_clustering(clustering, rows)

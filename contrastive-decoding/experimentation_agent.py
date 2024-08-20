from assistant_find_divergence_prompts import DivergenceFinder
from topic_analyzer import TopicAnalyzer
from knowledge_base import KnowledgeBase
from ai_assistant import AIAssistant

class ExperimentationAgent:
    def __init__(self, model1, model2, topics, ai_model="gpt-4", **kwargs):
        self.divergence_finder = DivergenceFinder(
            model_name=model1,
            starting_model_path=model1,
            comparison_model_path=model2,
            generation_length=kwargs.get('generation_length', 40),
            n_cycles_ask_assistant=kwargs.get('n_cycles_ask_assistant', 5),
            ai_model=ai_model
        )
        self.topic_analyzer = TopicAnalyzer(topics)
        self.knowledge_base = KnowledgeBase()
        self.ai_assistant = AIAssistant(model=ai_model)

    def generate_hypotheses(self):
        context = f"Based on the following findings: {self.knowledge_base.get_summary()}, generate 3 hypotheses about potential differences between the models."
        return self.ai_assistant.generate(context)

    def design_experiment(self, hypothesis):
        context = f"Design an experiment to test the following hypothesis: {hypothesis}. Include specific prompts and expected outcomes."
        return self.ai_assistant.generate(context)

    def run_experiment(self, experiment):
        prompts = self.parse_experiment(experiment)
        return self.divergence_finder.find_divergences(prompts)

    def analyze_results(self, results):
        return self.divergence_finder.analyze_divergences(results)

    def update_knowledge_base(self, analysis):
        self.knowledge_base.update(analysis)

    def parse_experiment(self, experiment):
        # Implement parsing logic for the experiment design
        # This should extract the prompts from the experiment description
        return []  # Placeholder implementation
from experimentation_agent import ExperimentationAgent
from logger import Logger
from visualizer import Visualizer

class AutomatedPipeline:
    def __init__(self, model1, model2, topics, ai_model="gpt-4", **kwargs):
        self.experimentation_agent = ExperimentationAgent(model1, model2, topics, ai_model, **kwargs)
        self.logger = Logger()
        self.visualizer = Visualizer()

    def run(self, num_cycles=5):
        for cycle in range(num_cycles):
            hypotheses = self.experimentation_agent.generate_hypotheses()
            for hypothesis in hypotheses:
                experiment = self.experimentation_agent.design_experiment(hypothesis)
                results = self.experimentation_agent.run_experiment(experiment)
                analysis = self.experimentation_agent.analyze_results(results)
                self.experimentation_agent.update_knowledge_base(analysis)
                self.logger.log_experiment(experiment, results, analysis)
            self.visualizer.update_visualizations()
        
        return self.logger.generate_report()
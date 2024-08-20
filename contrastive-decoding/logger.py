import json
from datetime import datetime

class Logger:
    def __init__(self):
        self.log = []

    def log_experiment(self, experiment, results, analysis):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": experiment,
            "results": results,
            "analysis": analysis
        }
        self.log.append(entry)

    def generate_report(self):
        report = {
            "total_experiments": len(self.log),
            "experiments": self.log
        }
        return json.dumps(report, indent=2)

    def save_report(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
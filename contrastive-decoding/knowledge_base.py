class KnowledgeBase:
    def __init__(self):
        self.findings = []

    def update(self, new_finding):
        self.findings.append(new_finding)

    def get_summary(self):
        return "\n".join(self.findings[-5:])  # Return the last 5 findings as a summary

    def get_all_findings(self):
        return self.findings
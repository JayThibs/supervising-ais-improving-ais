from gpt4_interface import GPT4Interface

class TopicAnalyzer:
    def __init__(self, topics):
        self.topics = topics
        self.gpt4_interface = GPT4Interface()

    def assess_relevance(self, text):
        context = f"Assess the relevance of the following text to these topics: {', '.join(self.topics)}. Provide a relevance score between 0 and 1."
        response = self.gpt4_interface.generate(context + f"\n\nText: {text}")
        return self.parse_relevance_score(response)

    def summarize_findings(self, texts):
        context = f"Summarize the key findings related to these topics: {', '.join(self.topics)}. Base your summary on the following texts:"
        for i, text in enumerate(texts):
            context += f"\n\nText {i+1}: {text}"
        return self.gpt4_interface.generate(context)

    def parse_relevance_score(self, response):
        # Implement parsing logic to extract the relevance score from the GPT-4 response
        # This is a placeholder implementation
        try:
            return float(response.strip())
        except ValueError:
            print(f"Error parsing relevance score from response: {response}")
            return 0.0
import openai


class Model:
    def __init__(self, name):
        self.name = name

    def generate(self, prompt):
        raise NotImplementedError


class GPT3(Model):
    def __init__(self):
        super().__init__("gpt3")

    def generate(self, prompt):
        response = openai.Completion.create(engine="davinci", prompt=prompt)
        return response.choices[0].text.strip()


class LLM(Model):
    def __init__(self):
        super().__init__("llm")

    def generate(self, prompt):
        response = other_api.query(prompt)
        return response

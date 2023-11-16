from mypackage.models import GPT3, LLM


def test_gpt3_generate():
    gpt3 = GPT3()
    assert gpt3.name == "gpt3"

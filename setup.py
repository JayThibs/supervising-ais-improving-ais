from setuptools import setup, find_packages

setup(
    name="behavioural_clustering",
    version="0.1",
    packages=find_packages(),
    install_requires=["openai", "langchain"],  # dependencies
)

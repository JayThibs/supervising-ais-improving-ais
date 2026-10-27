from setuptools import setup, find_packages

setup(
    name="soft-prompting",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "generate-hard-prompts=soft_prompting.scripts.generate_hard_prompts:main",
            "train-soft-prompts=soft_prompting.scripts.train_divergence_soft_prompts:main",
            "evaluate-hard-prompts=soft_prompting.scripts.evaluate_hard_prompts:main",
        ],
    },
)

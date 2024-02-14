# Supervising AIs Improving AIs

This repository contains methods for evaluating the behavioral effects of different interventions (finetuning, knowledge editing, etc) on large language models by comparing the behavior of the starting model to the behavior of the model produced by the intervention. It also contains code for applying those methods to evaluate the safety of different interventions.

The rise of powerful, general language modeling systems has lead to the rise of a bewildering variety of techniques and datasets for influencing the behaviors of those systems. However, it's often difficult to tell how a given intervention actually changes a model's behavior. This makes it difficult to evaluate the safety of a given intervention, or to compare the safety of different interventions. Additionally, future prosaic AIs will likely shape their own development or that of successor AIs. This makes it important that we be able to quantify the impacts of whatever modifications they propose to make to themselves or their successors. 

# Table of Contents

- [Summary](#summary)
- [Projects for the Agenda](#projects-for-the-agenda)
- [Methods](#methods)
  - [Contrastive Decoding](#contrastive-decoding)
  - [Clustering](#clustering)
  - [LLM-Automated Analysis](#llm-automated-analysis)
- [How to Use](#how-to-use)
  - [Setup](#setup)
  - [Running](#running)
- [TODO](#todo)
- [Acknowledgements](#acknowledgements)

# Summary

There are two main ways AIs can get better: by improving their training algorithms or by improving their training data.

We tentatively believe that data-based improvement is riskier than architecture based improvement. Current models mostly derive their behavior from their training data, and not training algorithms (meaning their architectures, hyperparameters, loss functions, optimizers or the like). So far, most improvements to AI training algorithms seem 'value neutral'. Additionally, most of human value drift currently derives from cultural shifts changing the 'training data' available in the environment, not biological evolution over the brain's base learning algorithms.

We imagine a future where AIs self-augment by continuously seeking out more and better training data, and either creating successor AIs or training themselves on that data. Often, these data will come from the AIs running experiments in the real world (doing science), deliberately seeking data that would cover a specific gap in its current capabilities, analogous to how human scientists seek data from domains where our current understanding is limited. With AI, this could involve AgentGPT-like systems that spin up many instances of themselves to run experiments in parallel, potentially leading to quick improvements if we are in an agency overhang.

We want to find methods of ensuring such 'automated science' processes remain safe and controllable, even after many rounds of self-directed data collection and training. In particular, we consider problems such as:

* Preventing self-training from amplifying undesirable behaviors
* Preventing semantic drift in concept representations during self-training
* Ensuring cross-modality actions (such as a generated image for a text-to-image model, or robot movement for a text-and-image-to-actuator-motion model) remain grounded in their natural language descriptions after self-training in a non-lingual modality
* Preventing value drift during multiple, iterated steps of self-retraining

# Projects for the Agenda

Currently, we're focusing on scalable methods of tracking behavioral drift in language models. In the near future, we will start working on benchmarks for evaluating a language model's capacity for stable self-modification via influencing its own training data.

# Methods

## Contrastive Decoding

Contrastive decoding is a method for comparing the behavioral tendencies of two language models. It works by querying the two models to generate a wide variety of responses, then using a combination of unsupervised clustering and supervisor models to compare the response patterns of the two LMs, and automatically highlight any differences that seems surprising or relevant from an alignment perspective.

## Clustering

Clustering is a method for grouping generated responses into clusters based on their semantic similarity. This is useful for identifying the different types of responses a model can generate, and for identifying which types of responses are most common.

## LLM-Automated Analysis

LLM-Automated Analysis involves using a language model to evaluate the behavioral tendencies of another language model. We use this approach along with the clustering techniques to study the behavioral differences between language models. Primarily, we measure the behavioral differences between a pre-trained language model and its fine-tuned version.

# How to Use

## Setup

1. Clone the repository

```bash
git clone https://github.com/JayThibs/supervising-ais-improving-ais
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Download the models

```bash
python download_models.py
```

For models the Llama-2, you will need to request access on the Meta website.

## Running

1. Run the following command to generate an entire test run:
    
```bash
python unsupervised_llm_behavioural_clustering/main.py --model_family="openai" --model="gpt-3.5-turbo" --test_mode
```

2. Run the following command to do a full run on 5000 statements:

```bash
python unsupervised_llm_behavioural_clustering/main.py --model_family="openai" --model="gpt-3.5-turbo" --num_statements=5000
```

# TODO

* Refactor notebook code in python scripts
* Integrate all methods
* Improve clustering method
* Improve labeling method

# Acknowledgements

This work is supported through grants by the [Long-Term Future Fund](https://funds.effectivealtruism.org/funds/far-future), [Lightspeed Grants](https://lightspeedgrants.org/), and [Open Philanthropy](https://www.openphilanthropy.org/), and partially took place at [Cavendish Labs](https://cavendishlabs.org/).

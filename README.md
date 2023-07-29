# Supervising AIs Improving AIs

Future prosaic AIs will likely shape their own development or that of successor AIs. We're trying to make sure they don't go insane.

# Summary

There are two main ways AIs can get better: by improving their training algorithms or by improving their training data. 

We consider both scenarios, and tentatively believe that data-based improvement is riskier than architecture based improvement. Current models mostly derive their behavior from their training data, and not training algorithms (meaning their architectures, hyperparameters, loss functions, optimizers or the like)[1]. So far, most improvements to AI training algorithms seem 'value neutral'[2]. Also note that most of human value drift currently derives from cultural shifts changing the 'training data' available in the environment, not biological evolution over the brain's base learning algorithms[3]. 

We imagine a future where AIs self-augment by continuously seeking out more and better training data, and either creating successor AIs or training themselves on that data. Often, these data will come from the AIs running experiments in the real world (doing science), deliberately seeking data that would cover a specific gap in its current capabilities, analogous to how human scientists seek data from domains where our current understanding is limited. With AI, this could involve AgentGPT-like systems that spin up many instances of themselves to run experiments in parallel, potentially leading to quick improvements if we are in an agency overhang.

We want to find methods of ensuring such 'automated science' processes remain safe and controllable, even after many rounds of self-directed data collection and training. In particular, we consider problems such as:

    Preventing self-training from amplifying undesirable behaviors
    Preventing semantic drift in concept representations during self-training
    Ensuring cross-modality actions (such as a generated image for a text-to-image model, or robot movement for a text-and-image-to-actuator-motion model) remain grounded in their natural language descriptions after self-training in a non-lingual modality
    Preventing value drift during multiple, iterated steps of self-retraining

# Projects for the Agenda

Currently, we're focusing on scalable methods of tracking behavioral drift in language models, as well as benchmarks for evaluating a language model's capacity for stable self-modification via self-training.

We now have a channel on the EleutherAI discord server called ai-supervisors. If you’d like to help with this agenda, please go there!

In the channel, Quintin shared a quick overview of the two projects we mentioned in this post. I’m sharing it below two provide some clarity on what we are working towards at the moment:

This agenda has two projects as its current focuses.

Project 1: Unsupervised behavioral evaluation
This project focuses on scalable ways to compare the behavioral tendencies of different LMs (or different ways of prompting the same LM), without necessarily knowing what you're looking for beforehand. The project's current approach is to query the two LMs to generate a wide variety of responses, then use a combination of unsupervised clustering and supervisor models to compare the response patterns of the two LMs, and automatically highlight any differences that seems surprising or relevant from an alignment perspective.

The ultimate goal of this project is to greatly accelerate the part of LM alignment research where we evaluate how a given finetuning / alignment approach impacts an LM's behaviors, so that LM alignment researchers can more quickly experiment with different finetuning approaches.

Project 2: Benchmarks for stable reflectivity
This project focuses on building probing datasets to evaluate a model's competence at various sub-tasks associated with reflectivity / metacognition / values stability. Currently, these sub-tasks include:
- Tracking one’s own values versus the values of others
- Differentiating one’s current values versus one’s future values
- Identifying events that could influence personal or others' values
- Predicting how events may impact one's values
- Evaluating the desirability of specific influences on personal values

Our intent is to generate ~300 high quality labeled data points for each subtask, as well as a pipeline for quickly generating and validating more such probing datasets.

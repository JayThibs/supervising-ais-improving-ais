Code for intervening on language models in various ways, so that we can then use our framework to analyze the impacts of those interventions.

Possible interventions to include:

* Autoregressive finetuning on a file containing texts.
* Knowledge editing
* Quantization
* Anti-sycophancy synthetic data: https://arxiv.org/abs/2308.03958
* Training the model to fill in the middle
* Ablating specific attention heads (and see if observed changes in model behavior match interpretability results for those heads)
* Training them to classify images / process other modalities
* Comparing distilled and non-distilled models
* Steering vector additions
* Soft prompt tuning
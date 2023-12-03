from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

dataset = load_dataset('wikitext', 'wikitext-2-v1')
tok = AutoTokenizer.from_pretrained("gpt2")

ds = [s.replace(" \'", "").replace(" ,", ",").replace(" .", ".").replace("\n", "\\n") for s in dataset['train']['text']]
ds = [s[1:] if len(s) > 1 and s[0] == ' ' else s for s in ds]

sents = []
for s in ds:
    sents += [v + '.\n' for v in s.split(". ") if len(tok.tokenize(v)) > 10]
rand_index = np.random.permutation(range(len(sents)))
sents = [sents[i]  for i in rand_index]
print(len(sents))
with open("wikitext-2-v1-prompts.txt", "w") as f:
    f.writelines(sents)

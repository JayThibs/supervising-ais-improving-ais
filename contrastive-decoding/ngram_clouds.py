import pandas as pd
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
from wordcloud import WordCloud
from transformers import AutoTokenizer
import sys
sys.path.append("outputs")
from outputs.analysis_helpers import literal_eval_fallback
import numpy as np

# Given a list of text tokens, a list of divergences, and value n, this function finds each n-token long ngram in text and
# counts their total divergence score.
def compute_ngram_statistics(texts, divs, n):
    # Initialize a Counter object to hold the ngram counts
    ngram_counts = Counter()

    # Iterate over each text
    for tokens, token_divs in zip(texts, divs):
        
        if len(tokens) > 2:
            # First, find the ngrams in the current tokens, and the corresponding token indicies where they appear
            # Each time a given ngram appears in the current tokens, sum the divergences of the tokens that compose the ngram
            # and update the ngram count with the summed divergence
            
            # Generate ngrams for the current set of tokens
            current_ngrams = list(ngrams(tokens, n))
            # Iterate over each ngram
            for ngram in current_ngrams:
                # Find all start indices of this ngram in the tokens list
                start_indices = [i for i in range(len(tokens) - n + 1) if tokens[i:i+n] == list(ngram)]
                # Calculate the sum of divergences for the tokens in this ngram for each occurrence
                total_divergence = sum(sum(token_divs[i:i+n]) for i in start_indices)
                # Update the ngram count in the Counter object
                ngram_counts[ngram] += total_divergence
    # filter out entries with zero divergence
    ngram_counts = {ngram: count for ngram, count in ngram_counts.items() if count > 0}
            
    ngram_counts = {' '.join(ngram): count for ngram, count in ngram_counts.items()}
    # Return the ngram counts
    return ngram_counts




parser = argparse.ArgumentParser()
# Read path to pandas file with contrastively decoded texts
parser.add_argument("--path", type=str, default="", help="Path to pandas file with contrastively decoded texts")
# Read n value for ngram word cloud
parser.add_argument("--n", type=int, default=1, help="Value of n for ngram word cloud")
parser.add_argument("--n_range", type=int, default=None, help="Runs ngram word cloud for n values in the range of 1 to n_range")
# Read the save location
parser.add_argument("--save_loc", type=str, default="wordcloud_ngram.png", help="Path to save the ngram cloud")
parser.add_argument("--tokenizer", type=str, default="NousResearch/Meta-Llama-3-8B", help="Tokenizer to use split the text into tokens")
parser.add_argument("--max_lines", type=int, default=10000, help="Maximum number of lines to read from the pandas file")
parser.add_argument("--dict_key_index", type=int, default=0, help="Which of the dictionary's keys to access for the current dataset")
parser.add_argument("--ignore_divs", action="store_true", default=False, help="Ignore the divergence scores and just use the token counts")

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

if args.path.endswith(".txt") or args.path.endswith(".csv") or args.path.endswith(".tsv"):
    # Read pandas file
    df = pd.read_csv(args.path)
    df = df.sample(n=min(args.max_lines, len(df)))
    texts = df['decoding'].values
    divs_0 = literal_eval_fallback(df['all_token_divergences'].values[0], None)
    texts_len = len(divs_0)
    divs = [literal_eval_fallback(s, [0.0] * texts_len) for s in df['all_token_divergences'].values]
    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
elif args.path.endswith(".pkl"):
    # Otherwise, we're reading a pickled dictionary
    dict_ = pd.read_pickle(args.path)
    keys = list(dict_.keys())
    print(f"Current key selected: {keys[args.dict_key_index]}")

    tokenized_texts = dict_[keys[args.dict_key_index]]['input_tokens']
    divs = dict_[keys[args.dict_key_index]]['forward_token_divergences']
    if len(tokenized_texts) > args.max_lines:
        selection = np.random.choice(len(tokenized_texts), args.max_lines, replace=False)
        tokenized_texts = [tokenized_texts[i] for i in selection]
        divs = [divs[i] for i in selection]
else:
    raise ValueError(f"Invalid file type for --path: {args.path}. Must be a pandas file or a pickled dictionary.")

if args.ignore_divs:
    divs = [[1.0 for _ in x] for x in divs]

# Strip away special tokenization characters (such as space indicators) from the token strings
tokenized_texts = [[tokenizer.convert_tokens_to_string([token]).replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r") for token in tokens] for tokens in tokenized_texts]


if args.n_range is not None:
    for n in range(1, args.n_range + 1):
        print(f"Computing ngram statistics for n={n}")
        # Compute ngram statistics
        ngram_counts = compute_ngram_statistics(tokenized_texts, divs, n)
        # Generate the wordcloud
        wc = WordCloud(width=2200, height=1600, max_words=150, background_color='white').generate_from_frequencies(ngram_counts)
        # Save the wordcloud
        plt.figure(figsize=(20, 16))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(args.save_loc.replace("ngram", str(n) + "gram"))
else:
    print(f"Computing ngram statistics for n={args.n}")
    # Compute ngram statistics
    ngram_counts = compute_ngram_statistics(tokenized_texts, divs, args.n)
    # Generate the wordcloud
    wc = WordCloud(width=2200, height=1600, max_words=150, background_color='white').generate_from_frequencies(ngram_counts)
    # Save the wordcloud
    plt.figure(figsize=(20, 16))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(args.save_loc.replace("ngram", str(args.n) + "gram"))


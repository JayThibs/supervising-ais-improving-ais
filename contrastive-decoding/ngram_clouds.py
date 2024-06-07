import pandas as pd
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
from wordcloud import WordCloud


# Given a list of texts and value n, this function finds each n-word long ngram in text and
# counts their frequency.
def compute_ngram_statistics(texts, n):
    # Initialize a Counter object to hold the ngram counts
    ngram_counts = Counter()

    # Iterate over each text
    for text in texts:
        # Tokenize the text into words
        words = text.split()
        if len(words) > 2:
            # Compute the ngrams for this text
            text_ngrams = list(ngrams(words, n))
            # Update the Counter with the ngrams from this text
            ngram_counts.update(text_ngrams)
    ngram_counts = {" ".join(ngram): count for ngram, count in ngram_counts.items()}
    # Return the ngram counts
    return ngram_counts


parser = argparse.ArgumentParser()
# Read path to pandas file with contrastively decoded texts
parser.add_argument(
    "--path",
    type=str,
    default="",
    help="Path to pandas file with contrastively decoded texts",
)
# Read n value for ngram word cloud
parser.add_argument("--n", type=int, default=1, help="Value of n for ngram word cloud")
# Read the save location
parser.add_argument(
    "--save_loc",
    type=str,
    default="wordcloud_ngram.png",
    help="Path to save the ngram cloud",
)

args = parser.parse_args()

# Read pandas file
df = pd.read_csv(args.path)
texts = df["decoding"].values

# Compute ngram statistics
ngram_counts = compute_ngram_statistics(texts, args.n)

# Use wordcloud and matplotlib to create and save an ngram wordcloud

# Generate the wordcloud
wc = WordCloud(
    width=1600, height=1200, max_words=300, background_color="white"
).generate_from_frequencies(ngram_counts)

# Save the wordcloud
plt.figure(figsize=(16, 12))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig(args.save_loc.replace("ngram", str(args.n) + "gram"))

# Print the ngram counts

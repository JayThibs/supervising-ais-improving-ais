import os
import sys
import json
import time
import random
import argparse
import pickle
import tqdm
import matplotlib.pyplot as plt
import scipy
import sklearn
import glob
import pandas as pd
import numpy as np
import openai
import langchain
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, OPTICS
from terminaltables import AsciiTable, SingleTable
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import helper_functions as hf

# Set up OpenAI API key from .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Anthropic evals dataset if not already downloaded
if not os.path.exists("data/evals"):
    # git clone the evals repo into the data folder
    os.system("git clone https://github.com/anthropics/evals.git data/anthropic_evals")

n_points = 5000
file_paths = [path for path in glob.iglob("evals/**/*.jsonl", recursive=True)]
all_texts = []
for path in file_paths:
    with open(path, "r") as f:
        json_lines = f.readlines()
        dicts = [[path, json.loads(l)] for l in json_lines]
        for d in dicts:
            if "statement" in d[1]:
                all_texts.append([d[0], d[1]["statement"]])
            # if "question" in d[1]:
            #    all_texts.append([d[0], d[1]['question']])

short_texts = [t[1] for t in all_texts if len(t[1]) < 150]
rng = np.random.default_rng(seed=42)
texts_subset = rng.permutation(short_texts)[:n_points]

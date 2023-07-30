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

# Set up OpenAI API key from .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

# Download Anthropic evals dataset if not already downloaded
if not os.path.exists("data/evals"):
    # git clone the evals repo into the data folder
    os.system("git clone https://github.com/anthropics/evals.git data/anthropic_evals")

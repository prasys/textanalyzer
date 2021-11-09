#https://www.kaggle.com/ceshine/tag-visualization-with-universal-sentence-encoder
import os
import re
import html as ihtml
import warnings
import random
warnings.filterwarnings('ignore')

os.environ["TFHUB_CACHE_DIR"] = "/tmp/"

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import scipy
import umap

import tensorflow as tf
import tensorflow_hub as hub

import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', -1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

%matplotlib inline



# np load the embeddings for it

embeddings_train = np.load('USE_tuned_alta_train.npy',allow_pickle=True)
#embeddings_test = np.load('USE_tuned_alta_test.npy',allow_pickle=True)
#sentence_embeddings = np.concatenate(embeddings_train,embeddings_test,axis=0)
sentence_embeddings = embeddings_train

# viusalize how well they look together , it should be similiar to all , probably concat them together (I guess)

embedding = umap.UMAP(metric="cosine", n_components=2, random_state=42).fit_transform(sentence_embeddings)

df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    df_emb_sample["x"].values, df_emb_sample["y"].values, s=1
)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)
plt.show()
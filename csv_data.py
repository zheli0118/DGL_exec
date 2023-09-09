import urllib.request

import pandas as pd

urllib.request.urlretrieve(
    "https://data.dgl.ai/tutorial/dataset/members.csv", "own_dataset/members.csv"
)
urllib.request.urlretrieve(
    "https://data.dgl.ai/tutorial/dataset/interactions.csv",
    "own_dataset/interactions.csv",
)

members = pd.read_csv("own_dataset/members.csv")
members.head()

interactions = pd.read_csv("own_dataset/interactions.csv")
interactions.head()
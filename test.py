import pandas as pd
edges = pd.read_csv("./own_dataset/graph_edges.csv")
edges_group = edges.groupby("graph_id")
print(edges_group.groups)
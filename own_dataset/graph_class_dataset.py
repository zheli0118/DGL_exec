import dgl
import urllib.request

import torch
from dgl.data import DGLDataset
import pandas as pd
import os

os.environ["DGLBACKEND"] = "pytorch"


class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="synthetic")

    def process(self):
        edges = pd.read_csv("./graph_edges.csv")
        properties = pd.read_csv("./graph_properties.csv")
        self.graphs = []
        self.labels = []

        label_dict = {}
        num_nodes_dict = {}
        # 初始化 label 和 nodes_dict  的信息
        for _, row in properties.iterrows():
            label_dict[row["graph_id"]] = row["label"]
            num_nodes_dict[row["graph_id"]] = row["num_nodes"]
        edges_group = edges.groupby("graph_id")
        # 从字典中拿到相应的信息，然后构图
        for graph_id in edges_group.groups:
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)
        self.labels = torch.LongTensor(self.labels)
    def __getitem__(self, item):
        return self.graphs[item],self.labels[item]
    def __len__(self):
        return len(self.graphs)


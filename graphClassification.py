import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.nn import GraphConv
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

# Generate a synthetic dataset with 10000 graphs, ranging from 10 to 500 nodes.
dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_example = len(dataset)
num_train = int(num_example * 0.8)
train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_example))
train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=5, drop_last=False)
test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=5, drop_last=False)
it = iter(train_loader)
batch_1 = iter(it)
print(batch_1)
# 每个batch返回图和对应的标签
batched_graph, labels = batch_1
graphs = dgl.unbatch(batched_graph)


class GCN(nn.Module):
    def __init__(self, in_feat, out_feat, number_class):
        self.number_class = number_class
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv1 = GraphConv(in_feat, out_feat)
        self.conv2 = GraphConv(out_feat, number_class)

    def forward(self, g, h):
        h = self.conv1(g, h)
        h = nn.ReLU(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")


model = GCN(dataset.dim_nfeats, 20, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(20):
    for batch_graph, labels in train_loader:
        pred = model(batch_graph, batch_graph.ndata["attr"].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
num_correct = 0
num_test = 0
for batch_graph, labels in test_loader:
    pred = model(batch_graph, batch_graph.ndata["attr"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_test += len(labels)
print("Test accuracy:", num_correct / num_test)
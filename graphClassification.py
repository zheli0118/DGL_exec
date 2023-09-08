import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
# Generate a synthetic dataset with 10000 graphs, ranging from 10 to 500 nodes.
dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
num_example=len(dataset)
num_train=int(num_example*0.8)
train_sampler= SubsetRandomSampler(torch.arange(num_train))
test_sampler=SubsetRandomSampler(torch.arange(num_train,num_example))
train_loader=GraphDataLoader(dataset,sampler=train_sampler,batch_size=5,drop_last=False)
test_loader=GraphDataLoader(dataset,sampler=test_sampler,batch_size=5,drop_last=False)
it=iter(train_loader)
batch_1=iter(it)
print(batch_1)
# 每个batch返回图和对应的标签
batched_graph,labels=batch_1
graphs=dgl.unbatch(batched_graph)

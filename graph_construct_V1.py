import dgl
import torch

# DGLGraphs are always directed, If you want to handle undirected graphs,
# you may consider treating it as a bidirectional graph
g = dgl.graph(([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]), num_nodes=7)
print(g.edges())

# assign and retrieve node and edge features via ndata and edata interface.
g.ndata["x"] = torch.randn(7, 3)
g.ndata["y"] = torch.randn(7, 3, 4)
g.edata["x"] = torch.randn(5, 3)

print(g.ndata["x"])

# Querying Graph Structures
print(g.num_nodes())
print(g.num_edges())
print(g.out_degrees())

# Graph Transformations¶
# extracting a subgraph
subgraph1 = g.subgraph([1,2,5])
print(subgraph1.ndata[dgl.NID])

# add reverse edge
newg = dgl.add_reverse_edges(subgraph1)
print(newg.edges())

# save_graph
dgl.save_graphs("constructV1.dgl",[g,subgraph1])

# load_graph
(ga,gb),_ = dgl.load_graphs("constructV1.dgl")
print(ga.edges())
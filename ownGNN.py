import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGEConv(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.out_feat = out_feat
        self.in_feat = in_feat
        self.linear = nn.Linear(2*in_feat,out_feat)
    def forward(self,g,h):
        """
        :param g: input graph
        :param h: feature
        """
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(
                message_func=fn.copy_u("h","m"),
                reduce_func=fn.mean("m","h_N"),
            )
            h_N=g.ndata["h_N"]
            h_total= torch.cat([h,h_N],dim=1)
            return self.linear(h_total)
if __name__ == '__main__':
    (g1,g2),_=dgl.load_graphs("constructV1.dgl")
    with g2.local_scope():
        print("1")

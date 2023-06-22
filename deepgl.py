from dgl import DGLGraph
import dgl.function as fn
import torch
import torch.nn as nn



class GNN(nn.Module):
    def __init__(self,in_feat,out_feat,num_layers):
        super(GNN,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(GNNLayer(in_feat,out_feat))
        for i in range(num_layers-1):
            self.layers.append(GNNLayer(2*out_feat,out_feat))

        
    def forward(self,g,node_indices=None,edge_indices=None):
        feat=g.ndata['features']

        for i in range(len(self.layers)):
            feat=self.layers[i](g,feat)
            print("-----------{x}-------------".format(x=i))
            print(feat)

        return feat


class GNNLayer(nn.Module):
    def __init__(self,in_feat,out_feat):
        super().__init__()
        self.layer=nn.Linear(in_feat,out_feat)

    
    def forward(self,g,feat):
        g.send_and_recv(g.edges(),fn.copy_u('x', 'm'), fn.sum('m', 'h'))

        # g.recv(g.nodes())
            
        return g.ndata['features']



g=DGLGraph()
g.add_nodes(3)
g.add_edges([0,1,2,1],[0,1,2,0])
g.ndata['x']=torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
print(g.ndata)
print(g.edges())
print(g.nodes())
model=GNN(3,3,3)
print(model(g))

import torch as th
g = dgl.graph(([0, 1], [1, 2]))
g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
# Define the function for sending node features as messages.
def send_source(edges):
    return {'m': edges.src['x']}
# Sum the messages received and use this to replace the original node feature.
def simple_reduce(nodes):
    return {'x': nodes.mailbox['m'].sum(1)}
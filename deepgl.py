from dgl import DGLGraph
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
            feat

        return feat


class GNNLayer(nn.Module):
    def __init__(self,in_feat,out_feat):
        super().__init__()
        self.layer=nn.Linear(in_feat,out_feat)

    
    def forward(self,g,feat):
        g.send_and_rcv(g.edges(),self.layer)

        g.recv(g.nodes())
            
        return feat



g=DGLGraph()
g.add_nodes(3)
g.add_edges([0,1,2,1],[0,1,2,0])
g.ndata['features']=torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]])
print(g.ndata)
print(g.edges())
print(g.nodes())
model=GNN(3,3,3)
print(model(g))
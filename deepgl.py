from dgl import DGLGraph
import sys
import dgl.function as fn
import torch
import torch.nn as nn
import logging
logger=logging.getLogger().addHandler(logging.StreamHandler())
class GNN(nn.Module):
    def __init__(self,in_feat,out_feat,num_layers):
        super(GNN,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(GNNLayer(in_feat,out_feat))
        for i in range(num_layers-1):
            self.layers.append(GNNLayer(out_feat,out_feat))

        
    def forward(self,g,node_indices=None,edge_indices=None):
        g.ndata['h']=g.ndata['x']

        for i in range(len(self.layers)):
            print("-----------{x}-------------".format(x=i))
            print("Information before aggregation")
            print(g.ndata['h'])
            g.ndata['h']=self.layers[i](g)
            print("Infomation after aggregation")
            print(g.ndata['h'])
            

        return g.ndata['h']


class GNNLayer(nn.Module):
    def __init__(self,in_feat,out_feat):
        super().__init__()
        self.layer=GNNTransformer(in_feat,out_feat)

    
    def forward(self,g):
        # print("Infomation after aggregatio
        g.send_and_recv(g.edges(),self.layer, self.reduce_func)
        # print(g)
    
        return g.ndata['m']
    
    def reduce_func(self,nodes):
        print("=---------Printing mailbox----------")
        print(nodes.mailbox['m'])
        return {'m': torch.sum(nodes.mailbox['m'], dim=1)}
    
    

class GNNTransformer(nn.Module):
    def __init__(self,input_feat,out_feat):
        super().__init__()
        self.layer=nn.Linear(input_feat,out_feat)
    
    def forward(self,edges):
        transformed=self.layer(edges.src['h'])
        print("-----------Printed transformed nodes-------------")
        print(transformed)
        return {'m': transformed}


g=DGLGraph()
g.add_nodes(3)
g.add_edges([0,1,1],[1,0,2])
g.ndata['x']=torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1]],dtype=torch.float32)
print(g.ndata)
# print(g.edges())
# print(g.nodes())
model=GNN(4,4,3)
res=(model(g))
print("----------------Printing single forward pass------------")
print(res)

sg = g.subgraph(torch.tensor([0,1]))
print(sg.ndata['x'])
print(sg.edges())
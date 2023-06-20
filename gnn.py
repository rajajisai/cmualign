import torch_geometric
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data

class GNN(nn.Module):
    def __init__(self,in_feat,out_feat,num_neighbour_samples,num_negatives,num_layers,num_relations):
        super(GNN,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(GNNlayer(in_feat,out_feat))
        for i in range(num_layers-1):
            self.layers.append(GNNlayer(out_feat,out_feat,num_relations))

        
    def forward(self,g,node_indices=None,edge_indices=None):
        feat=g.x
        print(g.x)
        for i in range(len(self.layers)):
            g.x=self.layers[i](g)
            print("-----------{x}-------------".format(x=i))
            print(g.x)

        return g.x


class GNNlayer(MessagePassing):
    def __init__(self,in_feat,out_feat,num_relations):
        super().__init__(aggr='add')
        self.layer=nn.Linear(in_feat,out_feat,)




    
    def forward(self,g):
        feat=g.x
        edge_index=g.edge_index
        # print("check1")
        # print(edge_index)
        g.x=self.propagate(edge_index=edge_index,feat=feat)
        
        return g.x

    def message(self,edge_index,feat_i):
        # x_j has shape [E, out_channels]
        # print("----HERe__-")
        # print(feat)
        # print(edge_index)
        # Step 4: Normalize node features.
        return feat_i

x=torch.tensor([[4,1,1,7],[1,1,1,9],[1,1,1,10],[1,1,1,10]])
# x=torch.tensor([[0,1,1],[1,1,1]])

# print(x.shape)
# edge_index=torch.tensor([[0,0,0,1,0,1,2,3],[1,2,3,0,0,1,2,3]],dtype=torch.int64)
edge_index=torch.tensor([[0,1,2,0,1,1],[0,1,2,1,1,1]],dtype=torch.int64)

g=Data(x,edge_index)
print(g.num_node_features)
g.generate_ids()
print(g.is_directed())
print(g.n_id)
g.remove_edge_index(0)
print(g.e_id)

# print(g.x)
model=GNN(3,3,3)
# m=GNNlayer(3,3)
# print(m(g))
print(model(g))
        



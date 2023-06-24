import torch_geometric
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data

class GNN(nn.Module):
    def __init__(self,in_feat,out_feat,num_layers):
        super(GNN,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(GNNLayer(in_feat,out_feat))
        for i in range(num_layers-1):
            self.layers.append(GNNLayer(2*out_feat,out_feat))

        
    def forward(self,g,node_indices=None,edge_indices=None):
        print("Printing number of layers")
        feat=g.x
        # print(len(self.layers))
        for i in range(len(self.layers)):
            feat=self.layers[i](g,feat)
            print("-----------{x}-------------".format(x=i))
            print(feat)
            print(feat.shape)

        return feat


class GNNLayer(MessagePassing):
    def __init__(self,in_feat,out_feat):
        super().__init__(aggr='add')
        self.layer=nn.Linear(in_feat,out_feat,dtype=torch.float64)

    
    def forward(self,g,feat,edge_index=None):
        # print("-------In Forward------")
        edge_index=torch.tensor([[0,1],[2,0]])
        # print(feat)
        temp_feat=self.layer(feat)
        feat=torch.cat((temp_feat,temp_feat),dim=1)
        feat=self.propagate(edge_index=edge_index,feat=feat)
        
        return feat

    def message(self,edge_index,feat_j):
        # x_j has shape [E, out_channels]
        # print("----HERe__-")
        print("In message")
        print(feat_j)
        # print(feat)
        # print(edge_index)

        # Step 4: Normalize node features.
        return feat_j

class Transform(nn.Module):
    def __init__(self,feat_dim, output_dim, params = None, attn_param = None, attn = False):
        if(params==None):
            self.layer=nn.Linear(feat_dim,out_dim,bias=True)
        else:
            self.layer=params
        self.cross_attn=attn
        if not self.cross_attn:
            if attn_param:
                self.attn_fc = attn_param
            else:
                self.attn_fc = nn.Linear(2 * out_feats, 1, bias=True)
                
    def forward(self,edge_indices):
        if 0:
            pass

x=torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],dtype=torch.float64)
print(x.dtype)
# x=torch.tensor([[0,1,1],[1,1,1]])

# print(x.shape)
# edge_index=torch.tensor([[0,0,0,1,0,1,2,3],[1,2,3,0,0,1,2,3]],dtype=torch.int64)
edge_index=torch.tensor([[0,1,2,0],[0,1,2,1]],dtype=torch.float64)

g=Data(x,edge_index)
# print(g.num_node_features)
g.generate_ids()
# print(g.is_directed())
# print(g.n_id)
# g.remove_edge_index(0)
# print(g.e_id)

# print(g.x)
model=GNN(4,3,4)
# m=GNNlayer(3,3)
# print(m(g))
print(model(g))
        



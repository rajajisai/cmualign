from typing import Optional, Union
from torch_scatter import scatter
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor


def row_wise_sum(list_of_tensors):
    # Initialize the result list
    result = []

    # Iterate over each tensor in the list
    for tensor in list_of_tensors:
        # Perform the row-wise sum
        row_sum = torch.sum(tensor, dim=0)

        # Append the row sum to the result list
        result.append(row_sum)

    return result


class GNN(nn.Module):

    def __init__(self, in_feat, out_feat, num_relations, num_layers):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(in_feat, out_feat, num_relations))
        for i in range(num_layers-1):
            self.layers.append(GNNLayer(2*out_feat, out_feat, num_relations))
        self.hidden_dim = out_feat
        self.num_layers = num_layers
        self.num_relations = num_relations

    def forward(self, g, relation_edges_list=None):
        print("Number of layers", self.num_layers)
        # Extracting input features
        h = g.x

        for i in range(self.num_layers-1):
            print("-----------Layer {x}-------------".format(x=i))
            h = self.layers[i](g, h, self.hidden_dim, relation_edges_list, cross_attn=False)
        print("-----------Layer {x}-------------".format(x=len(self.layers)-1))
        h = self.layers[self.num_layers-1](g, h, self.hidden_dim, relation_edges_list, cross_attn=True)

        return h


class GNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_relations):
        super(GNNLayer, self).__init__()

        self.self_transformer = nn.Linear(in_feat, out_feat, dtype=torch.float64)
        self.relation_transformers = nn.ModuleList()
        self.attention_transformers = nn.ModuleList()
        for i in range(num_relations):
            self.relation_transformers.append(Transform(in_feat, out_feat, cross_attn=False))
            self.attention_transformers.append(Transform(in_feat, out_feat, cross_attn=True))

        self.num_relations = num_relations

    def forward(self, g, feat, out_feat, relation_edges_list, cross_attn=False):

        # Creating list for neighbhour messages and cross attention

        neighbour_messages_list = [torch.empty((0, out_feat)).cuda()]*feat.shape[0]
        cross_attn_messages_list = [[torch.empty((0, out_feat)).cuda()]*feat.shape[0]

        # Storing neighbourhood size

        neighbourhood_sizes = torch.zeros((feat.shape[0], 1), dtype=torch.float64).cuda()
        # Pushing messages along edges.Cross attention is calculated by passing messages along negative relation edges

        print("Propagating messages ")
        self_h = self.self_transformer(feat)
        z_neighbour_aggregate = torch.zeros((feat.shape[0], out_feat), dtype=torch.float64).cuda()
        if (cross_attn):
            print("Computing cross attention")
        for relation_id in range(self.num_relations):
            if not cross_attn:

                aggregate = self.relation_transformers[relation_id](
                    relation_edges_list[relation_id], feat, neighbourhood_sizes)
                z_neighbour_aggregate += aggregate

            else:
                # Cross attention computation is done here TODO

                self.relation_transformers[relation_id](
                    relation_edges_list[relation_id], feat, neighbourhood_sizes, messages_list=neighbour_messages_list, self_h=self_h, cross_attn=True)
                self.attention_transformers[relation_id](
                    relation_edges_list[relation_id+self.num_relations], feat, neighbourhood_sizes, messages_list=attn_messages_list[0], cross_attn=True)

        if cross_attn:
            z_neighbour_aggregate = torch.stack(row_wise_sum(neighbour_messages_list)).cuda()
            alpha = []
            for i, (neighbour_embedding, cross_embedding) in enumerate(zip(neighbour_messages_list, attn_messages_list[0])):
                if neighbour_embedding.nelement() == 0 or cross_embedding.nelement() == 0:
                    alpha.append(torch.zeros(1, out_feat))
                else:
                    cross_difference = neighbour_embedding.unsqueeze(1).repeat(
                        1, cross_embedding.shape[0], 1)-cross_embedding.reshape((1, cross_embedding.shape[0], -1))
                    # print(cross_difference)
                    print("PRININT NORM ", i)
                    # print(-torch.norm(cross_difference, p=2, dim=2).sum(1))
                    print(torch.softmax(-torch.norm(cross_difference, p=2, dim=2).sum(1), dim=0).shape)
                    alpha.append(torch.softmax(-torch.norm(cross_difference, p=2, dim=2).sum(1), dim=0))

        if (cross_attn):
            print(len(alpha))
            import pdb
            pdb.set_trace()

        neighbourhood_sizes[neighbourhood_sizes == 0] = 1

        z_neighbour_normalized = torch.div(z_neighbour_aggregate, neighbourhood_sizes)

        # Appending self transform with neighbouring nodes relation transform aggregation
        z = torch.cat((self_h, z_neighbour_normalized), dim=1)

        # Applying Mask to exclude nodes not receiving any messages in the current batch

        mask = torch.zeros((feat.shape[0], 1)).cuda()
        for edge_pairs in relation_edges_list:
            mask[edge_pairs[1]] = 1

        h = torch.sigmoid(z)
        h = torch.mul(z, mask)
        return h


class Transform(MessagePassing):
    def __init__(self, input_dim, output_dim, params=None, attn_param=None, cross_attn=False):
        super(Transform, self).__init__()
        self.linear_transform = nn.Linear(input_dim, output_dim, dtype=torch.float64)
        if cross_attn:
            self.attention = nn.Linear(input_dim, 1, dtype=torch.float64)

    def forward(self, edge_index, feat, neighbourhood_size, messages_list=None, self_h=None, cross_attn=False):

        return self.propagate(edge_index=edge_index, feat=feat,  neighbourhood_sizes=neighbourhood_size, self_h=self_h, cross_attn=cross_attn, messages_list=messages_list)

    def message(self, feat_j, feat_i, cross_attn=False):

        z = self.linear_transform(feat_j)

        # TODO Implement Self attention here
        if cross_attn:
            return z

        return z

    def aggregate(self, z, edge_index, feat, neighbourhood_sizes, cross_attn, messages_list):

        aggregate = None

        # Calculating cumulative neighbour hood size so far
        destination_indices, size_increase = torch.unique(edge_index[1], return_counts=True)
        neighbourhood_sizes[destination_indices] += torch.reshape(size_increase, (size_increase.shape[0], 1))

        # Returning sum aggregate
        if not cross_attn:
            aggregate = scatter(z, edge_index[1], dim=0, dim_size=feat.shape[0], reduce="sum").cuda()
        else:
            for source_node_feature, destination_node_index in zip(z, edge_index[1]):

                messages_list[destination_node_index] = torch.cat((messages_list[destination_node_index], torch.reshape(
                    source_node_feature, (1, source_node_feature.shape[0]))), dim=0)

        return aggregate

    def update(self, aggregate):
        return aggregate


x = torch.tensor(
    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [8, 4, 3, 133],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [2, 3, 4, 5]],
    dtype=torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# relation_edges_list = [torch.tensor([[0, 5, 0, 1, 1, 1, 7, 8], [4, 8, 2, 3, 1, 5, 8, 8]]), torch.tensor(
#     [[1, 6, 2, 3, 4, 8], [4, 8, 1, 1, 6, 7]]), torch.tensor([[2, 3, 7, 6], [4, 4, 8, 7]])]
# relation_edges_list = relation_edges_list + \
#     [torch.tensor([[5, 8, 7], [1, 1, 2]]), torch.tensor([[5, 6], [2, 1]]), torch.tensor([[7, 1], [1, 7]])]
if True:
    relation_edges_list = [torch.tensor([[0, 2, 5, 7, 7, 3], [1, 1, 6, 6, 1, 2]]), torch.tensor(
        [[3, 4, 4, 5, 8], [1, 1, 6, 6, 6]]), torch.tensor([[1, 1, 1, 1, 6, 6, 6], [0, 2, 3, 4, 5, 7, 8]])]
    relation_edges_list = relation_edges_list + \
        [torch.tensor([[6, 7, 8, 5], [1, 1, 1, 1]]), torch.tensor(
            [[1, 2, 3], [6, 6, 6]]), torch.tensor([[3, 5], [2, 2]])]
for i, _ in enumerate(relation_edges_list):
    relation_edges_list[i] = relation_edges_list[i].cuda()
# print("Printing edge list")
# print(relation_edges_list)
# print(len(relation_edges_list))
num_relations = 3
g = Data(x, relation_edges_list)
g.generate_ids()

model = GNN(in_feat=4, out_feat=3, num_relations=num_relations, num_layers=2)
model = model.to(device=device)
print(model)
g.x = g.x.to(device=device)
result = model(g, relation_edges_list=relation_edges_list)
print("FINAL EMBEDDINGS")
print(result)

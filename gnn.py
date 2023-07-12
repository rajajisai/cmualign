import pdb
from typing import Optional, Union
from torch_scatter import scatter
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor
from utils import *


class GNN(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_relations, num_layers):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(feat_dim, hidden_dim, num_relations))
        for i in range(num_layers-1):
            self.layers.append(GNNLayer(2*hidden_dim, hidden_dim, num_relations))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.fc = nn.Linear(2*hidden_dim, 1)

    def forward(self, g, relation_edges_list=None):
        # print("Number of layers", self.num_layers)
        # Extracting input features
        h = g.x.cuda()
        neighbourhood_sizes = calculate_neighbourhood_size(h.shape[0], relation_edges_list, self.num_relations)

        for i in range(self.num_layers-1):
            # print("-----------Layer {x}-------------".format(x=i))
            h = self.layers[i](h, self.hidden_dim, relation_edges_list, neighbourhood_sizes, cross_attn=False)

        # print("-----------Layer {x}-------------".format(x=len(self.layers)-1))
        h = self.layers[self.num_layers-1](h, self.hidden_dim, relation_edges_list,
                                           neighbourhood_sizes, cross_attn=True)

        return h


class GNNLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_relations):
        super(GNNLayer, self).__init__()

        self.self_transformer = nn.Linear(feat_dim, hidden_dim, dtype=torch.float32)
        self.relation_transformers = nn.ModuleList()
        self.attention_transformers = nn.ModuleList()
        for i in range(num_relations):
            self.relation_transformers.append(Transform(feat_dim, hidden_dim, cross_attn=False))
            self.attention_transformers.append(Transform(feat_dim, hidden_dim, cross_attn=True))
        self.num_relations = num_relations

    def forward(self, feat, hidden_dim, relation_edges_list, neighbourhood_sizes, cross_attn=False):

        # Creating list for neighbhour messages ,cross attention and self_attention
        mask = torch.zeros((feat.shape[0], 1)).cuda()
        for edge_pairs in relation_edges_list:
            mask[edge_pairs[1]] = 1

        neighbour_messages_list = [torch.empty((0, hidden_dim)).cuda()]*feat.shape[0]
        cross_attn_messages_list = [torch.empty((0, hidden_dim)).cuda()]*feat.shape[0]
        self_attn_list = [torch.empty((0, 1)).cuda()]*feat.shape[0]

        # Pushing messages along edges.Cross attention is calculated by passing messages along negative relation edges

        # print("Propagating messages ")
        self_h = self.self_transformer(feat)
        # print(self_h)
        z_neighbour_aggregate = torch.zeros((feat.shape[0], hidden_dim), dtype=torch.float32).cuda()

        for relation_id in range(self.num_relations):

            if not cross_attn and relation_edges_list[relation_id].nelement() != 0:

                z_neighbour_aggregate += self.relation_transformers[relation_id](
                    relation_edges_list[relation_id], feat)
                #  += aggregate

            elif relation_edges_list[relation_id].nelement() != 0:
                # Neighbour relation transform message
                self.relation_transformers[relation_id](
                    relation_edges_list[relation_id], feat, neighbour_messages_list, cross_attn=True)

                # cross attention message
                self.attention_transformers[relation_id](
                    relation_edges_list[relation_id+self.num_relations], feat, cross_attn_messages_list, cross_attn=True)

                # self attention message
                # print("Computing self attention for relation ", relation_id)
                # print(self_h)
                self.attention_transformers[relation_id](
                    relation_edges_list[relation_id], feat,  self_attn_list, self_h=self_h, cross_attn=True)

        if cross_attn:
            z_neighbour_aggregate = torch.stack(row_wise_sum(neighbour_messages_list)).cuda()
            alpha_list = []
            beta_list = []
            for i, (neighbour_embedding, cross_embedding, self_attention) in enumerate(zip(neighbour_messages_list, cross_attn_messages_list, self_attn_list)):
                if neighbour_embedding.nelement() == 0 or cross_embedding.nelement() == 0:
                    alpha_list.append(torch.zeros(1, 1).cuda())
                    beta_list.append(torch.zeros(1, 1).cuda())
                else:
                    cross_pair_difference = neighbour_embedding.unsqueeze(1).repeat(
                        1, cross_embedding.shape[0], 1)-cross_embedding.reshape((1, cross_embedding.shape[0], -1))

                    alpha_list.append(
                        torch.softmax(-torch.norm(cross_pair_difference, p=2, dim=2).sum(1), dim=0).reshape(-1, 1).cuda())

                    beta_list.append(torch.softmax(self_attention, dim=0).reshape(-1, 1))

            for i, (neighbour_embedding, alpha, beta) in enumerate(zip(neighbour_messages_list, alpha_list, beta_list)):

                temp_attn = neighbour_embedding*alpha*beta

                z_neighbour_aggregate[i] = temp_attn.sum(dim=0)
                z = torch.cat((self_h, z_neighbour_aggregate), dim=1)
                z = torch.mul(torch.sigmoid(z), mask)
                # z = torch.mul(z, mask)

            return z
        else:
            # Normalizing by neighbourhood size
            z_neighbour_aggregate = torch.div(z_neighbour_aggregate, neighbourhood_sizes)

        # Appending self transform with neighbouring nodes relation transform aggregation
        z = torch.cat((self_h, z_neighbour_aggregate), dim=1)

        # Applying Mask to exclude nodes not receiving any messages in the current batch

        h = torch.sigmoid(z)
        h = torch.mul(z, mask)
        return h


class Transform(MessagePassing):

    def __init__(self, feat_dim, output_dim, params=None, attn_param=None, cross_attn=False):
        super(Transform, self).__init__()

        self.linear_transform = nn.Linear(feat_dim, output_dim, dtype=torch.float32)
        self.attention = nn.Linear(feat_dim, 1, dtype=torch.float32)

    def forward(self, edge_index, feat, messages_list=None, self_h=None, cross_attn=False):

        return self.propagate(edge_index=edge_index, feat=feat, self_h=self_h, messages_list=messages_list, cross_attn=cross_attn)

    def message(self, feat_j, self_h_i, cross_attn=False):

        z = self.linear_transform(feat_j)

        if cross_attn:

            if (self_h_i is not None):

                z = self.attention(torch.cat((self_h_i, z), dim=1))

        return z

    def aggregate(self, z, edge_index, feat, cross_attn, messages_list):

        # Returning sum aggregate
        if not cross_attn:

            z = scatter(z, edge_index[1], dim=0, dim_size=feat.shape[0], reduce="sum").cuda()

        else:
            for source_node_feature, destination_node_index in zip(z, edge_index[1]):

                messages_list[destination_node_index] = torch.cat(
                    (messages_list[destination_node_index], torch.reshape(source_node_feature, (1, source_node_feature.shape[0]))), dim=0)

        return z


# x = torch.tensor(
#     [[1, 1, 1, 1],
#      [1, 1, 1, 1],
#      [8, 4, 3, 133],
#      [1, 1, 1, 1],
#      [1, 1, 1, 1],
#      [3, 4, 6, 2],
#      [1, 1, 1, 1],
#      [1, 1, 1, 1],
#      [2, 3, 4, 5]],
#     dtype=torch.float32)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # relation_edges_list = [torch.tensor([[0, 5, 0, 1, 1, 1, 7, 8], [4, 8, 2, 3, 1, 5, 8, 8]]), torch.tensor(
# #     [[1, 6, 2, 3, 4, 8], [4, 8, 1, 1, 6, 7]]), torch.tensor([[2, 3, 7, 6], [4, 4, 8, 7]])]
# # relation_edges_list = relation_edges_list + \
# #     [torch.tensor([[5, 8, 7], [1, 1, 2]]), torch.tensor([[5, 6], [2, 1]]), torch.tensor([[7, 1], [1, 7]])]
# if True:
#     relation_edges_list = [torch.tensor([[0, 2, 5, 7, 7, 3], [1, 1, 6, 6, 1, 2]]), torch.tensor(
#         [[3, 4, 4, 5, 8], [1, 1, 6, 6, 6]]), torch.tensor([[1, 1, 1, 1, 6, 6, 6], [0, 2, 3, 4, 5, 7, 8]])]
#     relation_edges_list = relation_edges_list + \
#         [torch.tensor([[6, 7, 8, 5], [1, 1, 1, 1]]), torch.tensor(
#             [[1, 2, 3], [6, 6, 6]]), torch.tensor([[3, 5], [2, 2]])]
# for i, _ in enumerate(relation_edges_list):
#     relation_edges_list[i] = relation_edges_list[i].cuda()
# calculate_neighbourhood_size(9, relation_edges_list, 3)

# num_relations = 3
# g = Data(x, relation_edges_list)
# g.generate_ids()

# model = GNN(feat_dim=4, hidden_dim=3, num_relations=num_relations, num_layers=2)
# model = model.to(device=device)
# print(model)
# g.x = g.x.to(device=device)
# result = model(g, relation_edges_list=relation_edges_list)
# print("FINAL EMBEDDINGS")
# print(result)

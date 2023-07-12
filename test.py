import torch
from gnn import GNN
from gnn import calculate_neighbourhood_size
from torch_geometric.data import Data

x = torch.tensor(
    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [8, 4, 3, 133],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [3, 4, 6, 2],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [2, 3, 4, 5]],
    dtype=torch.float32)

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
calculate_neighbourhood_size(9, relation_edges_list, 3)

num_relations = 3
g = Data(x, relation_edges_list)
g.generate_ids()

model = GNN(feat_dim=4, hidden_dim=3, num_relations=num_relations, num_layers=2)
model = model.to(device=device)
print(model)
g.x = g.x.to(device=device)
result = model(g, relation_edges_list=relation_edges_list)
print("FINAL EMBEDDINGS")
print(result)

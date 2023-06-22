import torch_geometric
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch_geometric.data import Data

# print(torch.gather(x, dim=0, index=index_dim0))
x=torch.tensor([[4,1,1,7],[1,1,1,9],[1,1,1,10],[2,1,1,10]])
indices=torch.tensor([0,1])
indices=[0,1]
print(x[indices])

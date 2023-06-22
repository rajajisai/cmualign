import dgl
import dgl.function as fn
import torch

g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
g.ndata['x'] = torch.ones(5, 2)
# Specify edges using (Tensor, Tensor).
g.send_and_recv(([1, 2], [2, 3]), fn.copy_u('x', 'm'), fn.sum('m', 'h'))
g.ndata['h']
# Specify edges using IDs.
g.send_and_recv([0, 2, 3], fn.copy_u('x', 'm'), fn.sum('m', 'h'))
g.ndata['h']

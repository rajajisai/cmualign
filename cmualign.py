from gnn import GNN
import torch
from preprocess import *
import torch.utils.data as tdata
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

g = torch.load("/home/rajaji/code/cmualign/amazon_wiki_pyg_1")
# print(g.x)
# train_data, val_data, test_data = generateTrainWithType(
#     'data/itunes_amazon_exp_data/exp_data/', graph_a, graph_b, positive_only=False)


def convert_relation_edge_list_to_tensor(relation_edge_list):
    # for i in relation_edge_list:
    #     relation_edge_list[i][0] = torch.tensor(relation_edge_list[i][0]).cuda()
    #     relation_edge_list[i][1] = torch.tensor(relation_edge_list[i][1]).cuda()
    for i, edge_index in enumerate(relation_edge_list):
        relation_edge_list[i] = torch.tensor(edge_index, dtype=torch.int64).cuda()
        # print(edge_index)


train_loader = tdata.DataLoader(
    g.train_data, batch_size=32, shuffle=True)
test_loader = tdata.DataLoader(
    g.test_data, batch_size=g.train_data.shape[0], shuffle=False)
val_loader = tdata.DataLoader(
    g.val_data, batch_size=g.val_data.shape[0], shuffle=False)

num_relations = 6
model = GNN(feat_dim=100, hidden_dim=50, num_relations=num_relations, num_layers=2)
model.cuda()

# Loss Function
loss_fcn = nn.BCEWithLogitsLoss()
# optimizer
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=1e-3, weight_decay=5e-4)

best_roc_score = 0
# for epoch in range(20):
#     model.train()
#     training_loss = 0.0
#     eids = []
#     offset = g.metadata["num_nodes_a"]
#     for batch in train_loader:
#         # print("HEllo")
#         relation_edge_list, test_nodes, eid = genEdgeBatch(
#             g, batch, offset, g.adj_a, g.adj_b, g.metadata["type_a_dict"], g.metadata["type_b_dict"], num_hops=1, num_neighbors=10)
#         convert_relation_edge_list_to_tensor(relation_edge_list)

#         # node_embeddings
#         h_final = model(g, relation_edge_list)

#         # loss caclulation and optimization
#         loss = loss_fcn(model.fc(h_final[batch[:, 0]]*h_final[batch[:, 1] + offset]
#                                  ).squeeze(), batch[:, 2].cuda().float())
#         training_loss += loss.detach().item()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(h_final.shape)

#     with torch.no_grad():
#         eids = []
#         for batch in val_loader:
#             # two graphs are concatenated
#             relation_edge_list, test_nodes, eid = genEdgeBatch(
#                 g, batch, offset, g.adj_a, g.adj_b, g.metadata["type_a_dict"], g.metadata["type_b_dict"], num_hops=1, num_neighbors=10)
#             convert_relation_edge_list_to_tensor(relation_edge_list)
#             h_final = model(g, relation_edge_list)
#             # emb = g.ndata['features']
#             score = model.fc(h_final[batch[:, 0]]*h_final[batch[:, 1] + offset]).squeeze()  # .sum(dim=1)
#             roc_score = roc_auc_score(batch[:, 2].numpy(), score.detach().cpu().numpy())
#             best_f1 = 0
#             for i in range(10):
#                 f1 = f1_score(batch[:, 2].numpy(), torch.sigmoid(score).detach().cpu().numpy() > 0.1 * i)
#                 best_f1 = max(best_f1, f1)
#                 # print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
#             # embed()
#             print('Validation AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))
#             if roc_score > best_roc_score:
#                 torch.save(model.state_dict(), 'best_gan.pkl')

offset = g.metadata["num_nodes_a"]
model.load_state_dict(torch.load('best_gan.pkl'))
with torch.no_grad():
    eids = []
    for batch in test_loader:
        # two graphs are concatenated
        relation_edge_list, test_nodes, eid = genEdgeBatch(
            g, batch, offset, g.adj_a, g.adj_b, g.metadata["type_a_dict"], g.metadata["type_b_dict"], num_hops=1, num_neighbors=10)
        convert_relation_edge_list_to_tensor(relation_edge_list)
        # print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
        emb = model(g, relation_edge_list)
        # emb = g.ndata['features']
        score = model.fc(emb[batch[:, 0]]*emb[batch[:, 1] + offset]).squeeze()  # .sum(dim=1)
        roc_score = roc_auc_score(batch[:, 2].numpy(), score.detach().cpu().numpy())
        best_f1 = 0
        for i in range(10):
            f1 = f1_score(batch[:, 2].numpy(), torch.sigmoid(score).detach().cpu().numpy() > 0.1 * i)
            best_f1 = max(best_f1, f1)
            # print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
        # embed()
        print('Test AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))

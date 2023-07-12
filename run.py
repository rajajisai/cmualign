import torch_geometric
import torch
import argparse
from models import module
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
from sklearn.metrics import roc_auc_score, f1_score
import random


def run(args):
    g = torch.load("/home/rajaji/code/cmualign/amazon_wiki_pyg_1")
    print("hello")
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)

    if args.model_opt == 0:
        print("---------------Using NICE HINGE-------------------")
        loss_fcn = module.NCE_HINGE()
    else:
        print("---------------Using BCE LOGITS-------------------")
        loss_fcn = nn.BCEWithLogitsLoss()

    model = module.BatchPairwiseDistance(p=2)
    in_feats = g.x.shape[0]
    num_rel = 3
    if args.gat == False:
        model_gan = module.smallGraphAlignNet(in_feats,
                                              g,
                                              args.num_negatives,
                                              args.n_hidden,
                                              args.n_layers,
                                              F.relu,
                                              args.dropout,
                                              num_rel,
                                              num_rel,
                                              args.model_opt,
                                              dist=model,
                                              loss_fcn=loss_fcn
                                              )
    if cuda:
        # g = g.cuda()
        model_gan.cuda()
        model.cuda()

    optimizer = torch.optim.Adam([{'params': model_gan.parameters()}],
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.validation:
        writer = SummaryWriter(comment=args.model_id + 'person_type')
        writer1 = SummaryWriter(comment=args.model_id + 'film_type')
    print(g.train_data)
    train_loader = tdata.DataLoader(g.train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = tdata.DataLoader(g.test_data, batch_size=g.train_data.shape[0], shuffle=False)
    val_loader = tdata.DataLoader(g.val_data, batch_size=g.val_data.shape[0], shuffle=False)
    best_roc_score = 0
    for epoch in range(args.n_epochs):
        model_gan.train()
        model.train()
        training_loss = 0.0
        eids = []
        offset = g.metadata["num_nodes_a"]
        for batch in train_loader:
            # two graphs are concatenated
            test_edges, test_nodes, eid = genEdgeBatch(
                g, batch, offset, g.adj_a, g.adj_b, g.metadata["type_a_dict"], g.metadata["type_b_dict"], num_hops=args.n_layers + 1, num_neighbors=args.num_neighbors)
            # print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
            eids += eid
            # print("-----------Printing TEST EDGES--------------")
            # print(test_edges)
            # print("-----------Printing TEST NODES--------------")
            # print(test_nodes)
            # print("Priting EIDS")
            # print(eid)
            # embed()
            emb = model_gan(g, test_edges, test_nodes)
            # print("---------------------Printing EMB------------------")
            # print(emb)
            # print(emb.shape)
            # embed()
            # print(g.ndata)
            if False:
                output_a, output_b = emb[batch[:, 0]].view(-1, args.num_negatives+1, 2 * args.n_hidden), emb[batch[:,
                                                                                                                   1] + offset].view(-1, args.num_test_negatives+1, 2 * args.n_hidden)
                # g.remove_edges(eid)
                logits = model(output_a, output_b)
                loss = loss_fcn(logits)
            else:
                # embed()
                # emb = g.ndata['features']
                # print("In ELSE")
                # print("Printing emb size")
                # print(emb[batch[:,0]].shape)
                # print((emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).shape)
                loss = loss_fcn(model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1] + offset]
                                             ).squeeze(), batch[:, 2].cuda().float())
            training_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(g.ndata)
            # print(g.ndata.keys())
        g.remove_edges(eids)
        del emb
        torch.cuda.empty_cache()

        print('Epoch:{}, loss:{}'.format(epoch, training_loss))
        # Validation
        with torch.no_grad():
            eids = []
            for batch in val_loader:
                # two graphs are concatenated
                test_edges, test_nodes, eid = genEdgeBatch(
                    g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops=args.n_layers + 1, num_neighbors=args.num_neighbors)
                # print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
                eids += eid
                emb = model_gan(g, test_edges, test_nodes)
                # emb = g.ndata['features']
                score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1] + offset]).squeeze()  # .sum(dim=1)
                roc_score = roc_auc_score(batch[:, 2].numpy(), score.detach().cpu().numpy())
                best_f1 = 0
                for i in range(10):
                    f1 = f1_score(batch[:, 2].numpy(), torch.sigmoid(score).detach().cpu().numpy() > 0.1 * i)
                    best_f1 = max(best_f1, f1)
                    # print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
                # embed()
                print('Validation AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))
                if roc_score > best_roc_score:
                    torch.save(model_gan.state_dict(), 'best_gan.pkl')

            g.remove_edges(eids)
        # Test
    model_gan.load_state_dict(torch.load('best_gan.pkl'))
    with torch.no_grad():
        eids = []
        for batch in test_loader:
            # two graphs are concatenated
            test_edges, test_nodes, eid = genEdgeBatch(
                g, batch, offset, adj_a, adj_b, type_a_dict, type_b_dict, num_hops=args.n_layers + 1, num_neighbors=args.num_neighbors)
            # print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
            eids += eid

            emb = model_gan(g, test_edges, test_nodes)
            # emb = g.ndata['features']
            score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1] + offset]).squeeze()  # .sum(dim=1)
            roc_score = roc_auc_score(batch[:, 2].numpy(), score.detach().cpu().numpy())
            best_f1 = 0
            for i in range(10):
                f1 = f1_score(batch[:, 2].numpy(), torch.sigmoid(score).detach().cpu().numpy() > 0.1 * i)
                best_f1 = max(best_f1, f1)
                # print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
            # embed()

            print('Test AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))

        g.remove_edges(eids)

    if args.validation:
        writer.close()
        writer1.close()
    pass


def genEdgeBatch(g, train_data, offset, adj_a, adj_b, type_a_dict, type_b_dict, add_edge=True, num_hops=1, num_neighbors=10, num_relations=3):
    train_data = train_data.numpy()
    # print("---------------------Printing Batch in GenEdge--------------------------")
    # print(train_data)
    nodes_a, nodes_b = set(train_data[:, 0].tolist()), set(
        train_data[:, 1].tolist())

    nodes = [list(nodes_a) + list(map(lambda x:x+offset, nodes_b))]

    edge_indices = defaultdict(list)
    eids = []

    left_nodes, right_nodes = set(), set()
    # print(type_a_dict)
    if True:
        for i in range(train_data.shape[0]):
            # Randomly sampling from nodes neighbourhood
            for n in random.sample(adj_a[train_data[i, 0]], min(num_neighbors, len(adj_a[train_data[i, 0]]))):
                left_nodes.add(n)
                for sub_edge in type_a_dict[(n, train_data[i, 0])]:

                    edge_indices[sub_edge].append(g.edge_id(n, train_data[i, 0]))

                # attn_edges.append(-type_a_dict[(n, train_data[i,0])] - 1)
                for sub_edge in type_a_dict[(n, train_data[i, 0])]:

                    edge_indices[sub_edge+2*num_relations].append(e_id)

            for m in random.sample(adj_b[train_data[i, 1]], min(num_neighbors, len(adj_b[train_data[i, 1]]))):
                right_nodes.add(m)
                for sub_edge in type_b_dict[(m, train_data[i, 1])]:
                    edge_indices[sub_edge + 1].append(
                        g.edge_id(m+offset, train_data[i, 1]+offset))
                    # print("------------------Printing Sub Edge Positive B-------------------")
                    # print(sub_edge)
                if add_edge:
                    # What are they doing here ??
                    g.add_edge(m+offset, train_data[i, 0])
                    # here is duplicate
                e_id = g.edge_id(m+offset, train_data[i, 0])

                # attn_edges.append(-type_b_dict[(m, train_data[i,1])] - 1)
                for sub_edge in type_b_dict[(m, train_data[i, 1])]:

                    edge_indices[sub_edge+2*num_relations].append(e_id)

    if num_hops > 8:
        # if False:
        nodes.append(list(left_nodes) +
                     list(map(lambda x: x+len(graph_a.id2idx), right_nodes)))
        for node_id in list(left_nodes):
            for n in random.sample(adj_a[node_id], min(num_neighbors, len(adj_a[node_id]))):
                for sub_edge in type_a_dict[(n, node_id)]:
                    try:
                        edge_indices[sub_edge + 1].append(g.edge_id(n, node_id))
                    except Exception as e:
                        # embed()
                        return e
        for node_id in list(right_nodes):
            for m in random.sample(adj_b[node_id], min(num_neighbors, len(adj_b[node_id]))):
                for sub_edge in type_b_dict[(m, node_id)]:
                    edge_indices[sub_edge + 1].append(
                        g.edge_id(m+len(graph_a.id2idx), node_id+len(graph_a.id2idx)))

    return edge_indices, nodes, eids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN')
    parser.add_argument("--preprocess", type=bool, default=False,
                        help="whether generate new gaph")
    parser.add_argument("--concat", type=bool, default=False,
                        help="whether concat at each hidden layer")

    parser.add_argument("--gat", action='store_true',
                        help="whether RGCN or RGAT is chosen")

    parser.add_argument("--model-opt", type=int, default=1,
                        help="[0: triplet loss, 1: binary classification]")

    parser.add_argument("--embedding", type=bool, default=False,
                        help="whether h0 is updated")
    parser.add_argument("--validation", type=bool, default=False,
                        help="whether draw pr-curve")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=10,
                        help="number of neighbors to be sampled")
    parser.add_argument("--num-negatives", type=int, default=10,
                        help="number of negative links to be sampled")
    parser.add_argument("--num-test-negatives", type=int, default=10,
                        help="number of negative links to be sampled in test setting")
    parser.add_argument("--n-hidden", type=int, default=50,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--burnin", type=int, default=-1,
                        help="when to use hard negatives")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--dump", action='store_true',
                        help="dump trained models (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--model-id", type=str,
                        help="Identifier of the current model")
    parser.add_argument("--pretrain_path", type=str, default="data/wiki.en.100.bin",
                        help="pretrained fastText path")
    args = parser.parse_args()

    print(args)

    run(args)

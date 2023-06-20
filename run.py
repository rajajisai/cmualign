import torch_geometric
import torch
import argparse
def run(args):
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
        #g = g.cuda()
        model_gan.cuda()
        model.cuda()

    optimizer = torch.optim.Adam([{'params': model_gan.parameters()}],
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    if args.validation:
        writer = SummaryWriter(comment=args.model_id + 'person_type')
        writer1 = SummaryWriter(comment=args.model_id + 'film_type')
    
    train_loader = tdata.DataLoader(g.train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = tdata.DataLoader(g.test_data, batch_size=g.train_data.shape[0], shuffle=False)
    val_loader = tdata.DataLoader(g.val_data, batch_size=g.val_data.shape[0], shuffle=False)
    best_roc_score = 0
    for epoch in range(args.n_epochs):
        model_gan.train()
        model.train()
        training_loss = 0.0
        eids = []
        for batch in train_loader:
            # two graphs are concatenated
            test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
            #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
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
                output_a, output_b = emb[batch[:, 0]].view(-1, args.num_negatives+1, 2 * args.n_hidden), emb[batch[:, 1] + offset].view(-1, args.num_test_negatives+1, 2 * args.n_hidden)
                #g.remove_edges(eid)
                logits = model(output_a, output_b)
                loss = loss_fcn(logits)
            else:
                # embed()
                # emb = g.ndata['features']
                # print("In ELSE")
                # print("Printing emb size")
                # print(emb[batch[:,0]].shape)
                # print((emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).shape)
                loss = loss_fcn( model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze(), batch[:, 2].cuda().float() )
            training_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(g.ndata)
            # print(g.ndata.keys())
        g.remove_edges(eids)
        del emb
        torch.cuda.empty_cache()
        
        print('Epoch:{}, loss:{}'.format(epoch, training_loss ))
        # Validation
        with torch.no_grad():
            eids = []
            for batch in val_loader:
                # two graphs are concatenated
                test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
                #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
                eids += eid
                emb = model_gan(g, test_edges, test_nodes)
                # emb = g.ndata['features']
                score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze() #.sum(dim=1)
                roc_score = roc_auc_score(batch[:,2].numpy(), score.detach().cpu().numpy())
                best_f1 = 0
                for i in range(10):
                    f1 = f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i ) 
                    best_f1 = max(best_f1, f1)
                    #print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
                # embed()
                print('Validation AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))
                if roc_score > best_roc_score:
                    torch.save(model_gan.state_dict(), 'best_gan.pkl')
                
            g.remove_edges(eids)
        #Test
    model_gan.load_state_dict(torch.load('best_gan.pkl'))
    with torch.no_grad():
            eids = []
            for batch in test_loader:
                # two graphs are concatenated
                test_edges, test_nodes, eid = genEdgeBatch(g, batch, graph_a, graph_b, adj_a, adj_b, type_a_dict, type_b_dict, num_hops = args.n_layers + 1, num_neighbors = args.num_neighbors)
                #print("Number of nodes:{}, Number of edges:{}".format(g.number_of_nodes(), g.number_of_edges()))
                eids += eid
                
                emb = model_gan(g, test_edges, test_nodes)
                # emb = g.ndata['features']
                score = model_gan.fc(emb[batch[:, 0]]*emb[batch[:, 1]+ offset]).squeeze() #.sum(dim=1)
                roc_score = roc_auc_score(batch[:,2].numpy(), score.detach().cpu().numpy())
                best_f1 = 0
                for i in range(10):
                    f1 = f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i ) 
                    best_f1 = max(best_f1, f1)
                    #print('ths:{}, f1:{}'.format(i, f1_score(batch[:,2].numpy(), torch.sigmoid(score).detach().cpu().numpy()>0.1 * i )))
                # embed()
                
                print('Test AUC_ROC:{}, Best F1:{}'.format(roc_score, best_f1))

            g.remove_edges(eids)
        
    if args.validation:
        writer.close()
        writer1.close()
    pass


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






















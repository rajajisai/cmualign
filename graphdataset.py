import torch_geometric
from torch_geometric.data import Data    
import torch 
from collections import defaultdict
from preprocess import Graph

def create_amazonwiki_graph(file_one=None,file_two=None):
    graph_a, graph_b = Graph("data/wiki.en100.bin"), Graph("data/wiki.en100.bin")
    graph_a.buildGraph('data/itunes_amazon_exp_data/exp_data/tableA.csv')
    graph_b.buildGraph('data/itunes_amazon_exp_data/exp_data/tableB.csv')
    
    #Displacing value node indices of graph_b by number of nodes in graph_a
    graph_b.edge_src=torch.add(graph_b.edge_src,len(graph_a.id2idx))
    graph_b.edge_dst=torch.add(graph_b.edge_dst,len(graph_a.id2idx))

    g=Data()
    # Appending edges to edge_index
    g.edge_index=torch.cat((graph_a.edge_src,graph_a.edge_dst),1)
    g.edge_index=torch.cat((g.edge_index,torch.cat((graph_a.edge_dst,graph_a.edge_src),1)),0)
    g.edge_index=torch.cat((g.edge_index,torch.cat((graph_b.edge_src,graph_b.edge_dst),1)),0)
    g.edge_index=torch.cat((g.edge_index,torch.cat((graph_b.edge_dst,graph_b.edge_src),1)),0)
    g.x= torch.cat([torch.FloatTensor(graph_a.features), torch.FloatTensor(graph_b.features)], 0).cuda()

    #Calculating metadata and adjacency list
    edge_type_a, edge_type_b = torch.LongTensor(graph_a.edge_type), torch.LongTensor(graph_b.edge_type)
    num_type_a, num_type_b = torch.max(edge_type_a).item() + 1, torch.max(edge_type_b).item() + 1
    type_a_dict, type_b_dict = defaultdict(list), defaultdict(list)
    adj_a, adj_b = defaultdict(list), defaultdict(list)
    for a,b,t in zip(graph_a.edge_src, graph_a.edge_dst, graph_a.edge_type):
        if b not in adj_a[a]:
            adj_a[a].append(b)
        if a not in adj_a[b]:
            adj_a[b].append(a)
        type_a_dict[(a,b)].append(t)
        type_a_dict[(b,a)].append(t + num_type_a)

    for a,b,t in zip(graph_b.edge_src, graph_b.edge_dst, graph_b.edge_type):
        if b not in adj_b[a]:
            adj_b[a].append(b)
        if a not in adj_b[b]:
            adj_b[b].append(a)
        type_b_dict[(a,b)].append(t)
        type_b_dict[(b,a)].append(t + num_type_b)
    train_data, val_data, test_data = generateTrainWithType('data/itunes_amazon_exp_data/exp_data/', graph_a, graph_b)
    # assume same number of relations 
    assert num_type_a == num_type_b
    #Adding to meta data to graph object
    g.metdata={}
    g.metdata["num_type_a"]=num_type_a
    g.metdata["num_type_b"]=num_type_b
    g.metdata["type_a_dict"]=type_a_dict
    g.metdata["type_b_dict"]=type_b_dict
    g.metdata["num_nodes_a"]=len(graph_a.id2idx)
    g.train_data=train_data
    g.val_data=val_data
    g.test_data=test_data
    print(g.keys)
    print(num_type_a)
    
    torch.save(g,"amazon_wiki_pyg")
create_amazonwiki_graph()
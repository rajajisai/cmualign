import sys
import pickle
from IPython import embed
from collections import defaultdict
import matplotlib.pyplot as plt
# from FeaturePipeline import FeatureGenerator, warp_feature
import numpy as np
from tqdm import tqdm
import fasttext as ft
import torch
import random  # from nltk.corpus import wordnet
import csv


class FeatureGenerator(object):
    """Name Embedding FeatureGenerator"""

    def __init__(self, model_path):
        super(FeatureGenerator, self).__init__()
        self.model = ft.load_model(model_path)

    def generateEmbFeature(self, name, sent=True):
        if sent == True:
            return self.model.get_sentence_vector(name.replace('"', ''))
        else:
            return self.model.get_word_vector(name.replace('"', ''))

# the interface to generate a numpy trix for every node in the graph


def warp_feature(model, source, mapping):
    print("--------------Warping Features-----------------")
    feature_matrix = np.zeros((len(source), 100))

    for k in source:

        for idx, attr in enumerate(source[k]):

            feature_matrix[mapping[k], :] += model.generateEmbFeature(attr, sent=True)

    print("----------------Printing Matrix Shape----------------------")
    print(feature_matrix.shape)
    return feature_matrix


def generateTrainWithType(in_path, graph_a, graph_b, positive_only=False):
    train_data, val_data, test_data = [], [], []
    with open(in_path+'train.csv') as IN:
        IN.readline()
        left_set, right_set = set(), set()
        for line in IN:
            tmp = line.strip().split(',')
            if tmp[0] not in left_set and tmp[1] not in right_set:
                left_set.add(tmp[0])
                right_set.add(tmp[1])
            else:
                continue

            # embed()
            train_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])],
                               graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])
    # embed()
    with open(in_path+'valid.csv') as IN:
        IN.readline()
        left_set, right_set = set(), set()
        for line in IN:
            tmp = line.strip().split(',')
            if tmp[0] not in left_set and tmp[1] not in right_set:
                left_set.add(tmp[0])
                right_set.add(tmp[1])
            else:
                continue
            # embed()
            val_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])],
                             graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])
    with open(in_path+'test.csv') as IN:
        IN.readline()
        left_set, right_set = set(), set()
        for line in IN:
            tmp = line.strip().split(',')
            if tmp[0] not in left_set and tmp[1] not in right_set:
                left_set.add(tmp[0])
                right_set.add(tmp[1])
            else:
                continue

            # embed()
            test_data.append([graph_a.id2idx['ID_{}'.format(tmp[0])],
                              graph_b.id2idx['ID_{}'.format(tmp[1])], int(tmp[2])])

    return torch.LongTensor(train_data), torch.LongTensor(val_data), torch.LongTensor(test_data)


def genEdgeBatch(g, train_data, offset, adj_a, adj_b, type_a_dict, type_b_dict, add_edge=True, num_hops=1, num_neighbors=10, num_relations=3):
    train_data = train_data.numpy()

    nodes_a, nodes_b = set(train_data[:, 0].tolist()), set(
        train_data[:, 1].tolist())

    nodes = [list(nodes_a) + list(map(lambda x:x+offset, nodes_b))]

    edge_indices = []
    eids = []

    left_nodes, right_nodes = set(), set()
    for relation_id in range(4*num_relations):
        edge_indices.append([[], []])
        # edge_indices[relation_id+2*num_relations] = [[], []]

    if True:
        for i in range(train_data.shape[0]):

            for n in random.sample(adj_a[train_data[i, 0]], min(num_neighbors, len(adj_a[train_data[i, 0]]))):

                left_nodes.add(n)

                for sub_edge in type_a_dict[(n, train_data[i, 0])]:

                    edge_indices[sub_edge][0].append(n)
                    edge_indices[sub_edge][1].append(train_data[i, 0])
                    if add_edge:
                        edge_indices[sub_edge+2*num_relations][0].append(n)
                        edge_indices[sub_edge+2*num_relations][1].append(train_data[i, 1]+offset)

            for m in random.sample(adj_b[train_data[i, 1]], min(num_neighbors, len(adj_b[train_data[i, 1]]))):
                right_nodes.add(m)
                for sub_edge in type_b_dict[(m, train_data[i, 1])]:
                    edge_indices[sub_edge][0].append(m+offset)
                    edge_indices[sub_edge][1].append(train_data[i, 1]+offset)
                    edge_indices[sub_edge+num_relations][0].append(train_data[i, 1]+offset)
                    edge_indices[sub_edge+num_relations][1].append(m+offset)
                    if add_edge:
                        edge_indices[sub_edge+2*num_relations][0].append(m+offset)
                        edge_indices[sub_edge+2*num_relations][1].append(train_data[i, 0])

    if num_hops > 8:

        nodes.append(list(left_nodes) +
                     list(map(lambda x: x+len(graph_a.id2idx), right_nodes)))
        for node_id in list(left_nodes):
            for n in random.sample(adj_a[node_id], min(num_neighbors, len(adj_a[node_id]))):
                for sub_edge in type_a_dict[(n, node_id)]:
                    try:
                        edge_indices[sub_edge + 1].append(g.edge_id(n, node_id))
                    except Exception as e:

                        return e
        for node_id in list(right_nodes):
            for m in random.sample(adj_b[node_id], min(num_neighbors, len(adj_b[node_id]))):
                for sub_edge in type_b_dict[(m, node_id)]:
                    edge_indices[sub_edge + 1].append(
                        g.edge_id(m+len(graph_a.id2idx), node_id+len(graph_a.id2idx)))
    #
    return edge_indices, nodes, eids


class Graph(object):
    """docstring for Graph"""

    def __init__(self, pretrain):
        super(Graph, self).__init__()
        # self.relation_list = relation_list
        self.id2idx = {}
        self.entity_table = {}
        self.features = None
        self.edge_src = []
        self.edge_dst = []
        self.edge_type = []
        self.pretrain_path = pretrain

    def buildGraph(self, table):
        # self.self.entity_table_table = self.entity_table_path
        #
        print("--------------------Building Graphs--------------------")
        with open(table, encoding="utf8") as IN:
            print("------------Reading CSV--------------")
            spamreader = csv.reader(IN, delimiter=',')
            # embed()
            # fields = IN.readline().strip().split(',')
            fields = next(spamreader)
            print(fields)
            type_list, type_dict = [], {}
            attr_list = []
            for idx, field in enumerate(fields[1:]):
                if '_' in field:
                    type_list.append(field.split('_')[0])
                else:
                    attr_list.append(field)
            print("-----------------Printing Type List---------------")
            print(type_list)
            print("--------------Printing Attribute List-----------")
            print(attr_list)
            edge_list = []

            for line in spamreader:

                tmp = line
                for idx, value in enumerate(tmp[1:]):

                    if idx < len(type_list):
                        if idx == 0:
                            _ID = 'ID_{}'.format(tmp[0])
                            self.entity_table[_ID] = [type_list[idx], value]
                            self.id2idx[_ID] = len(self.id2idx)
                            target_id = self.id2idx[_ID]
                        else:
                            _id = '{}_{}'.format(type_list[idx], value)

                            if _id not in self.entity_table:
                                self.entity_table[_id] = [type_list[idx], value]
                                # _ID = '{}_{}'.format(tm, type_list[idx])
                                self.id2idx[_id] = len(self.id2idx)
                            # edge_list.append([target_id, idx, id2idx[value]])
                            self.edge_src.append(target_id)
                            self.edge_dst.append(self.id2idx[_id])
                            self.edge_type.append(idx-1)
                            if (idx == 2):
                                self.edge_src.append(self.id2idx[type_list[1]+"_"+tmp[2]])
                                self.edge_dst.append(self.id2idx[_id])
                                self.edge_type.append(2)
                    else:
                        self.entity_table[_ID].append(value)

            feat = FeatureGenerator(self.pretrain_path)

            self.features = warp_feature(feat, self.entity_table, self.id2idx)
        # assert self.features.shape()[0] == len(self.id2idx)


def checkTest(mapping_a, mapping_b, in_file):
    type_cnt_a, type_cnt_b = defaultdict(int), defaultdict(int)
    str_pair = set()
    with open(in_file) as IN:
        for line in IN:
            tmp = line.strip().split('\t')
            if tmp[0] in mapping_a and tmp[1] in mapping_b:
                str_pair.add('{}_{}'.format(tmp[0], tmp[1]))
                for x in mapping_a[tmp[0]]['type']:
                    type_cnt_a[x] += 1
                for x in mapping_b[tmp[1]]['type']:
                    type_cnt_b[x] += 1
    print("Len of original data is {}".format(len(str_pair)))
    print(type_cnt_a, type_cnt_b)


if __name__ == '__main__':
    dataset = 'itunes'  # imdb
    graph_a, graph_b = Graph(), Graph()
    graph_a.buildGraph('data/itunes_amazon_exp_data/exp_data/tableA.csv')
    graph_b.buildGraph('data/itunes_amazon_exp_data/exp_data/tableB.csv')

import torch


def calculate_neighbourhood_size(num_nodes, relation_edges_list, num_relations):

    neighbourhood_sizes = torch.zeros((num_nodes, 1)).cuda()

    for relation_index, edge_index in enumerate(relation_edges_list):
        # print(edge_index)
        destination_indices, size_increase = torch.unique(edge_index[1], return_counts=True)
        neighbourhood_sizes[destination_indices] += torch.reshape(size_increase, (size_increase.shape[0], 1))
        if (relation_index == num_relations-1):
            break

    neighbourhood_sizes[neighbourhood_sizes == 0] = 1

    return neighbourhood_sizes.cuda()


def row_wise_sum(list_of_tensors):
    # Initialize the result list
    result = []

    # Iterate over each row of tensors in the list
    for tensor in list_of_tensors:

        # Perform the row-wise sum and append to result list
        row_sum = torch.sum(tensor, dim=0)
        result.append(row_sum)

    return result

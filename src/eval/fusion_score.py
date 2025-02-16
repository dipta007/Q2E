import warnings

import torch

from .utils import min_max_normalize

warnings.filterwarnings("ignore")


def scale(sim):
    return torch.round(sim * 10).int()


def entropy(p):
    return -torch.sum(p * torch.log2(p + 1e-6), dim=-1).unsqueeze(-1)


def fusion_inverse_entropy(args, results, params, queries, video_ids):
    sim_matrix = torch.zeros(len(queries), len(video_ids))
    for param, data in zip(params, results):
        curr_entropy = entropy(data)

        sim_matrix += (1 / (curr_entropy + 1e-6)) * data

    return min_max_normalize(sim_matrix)


def fusion_exp_entropy(args, results, params, queries, video_ids):
    sim_matrix = torch.zeros(len(queries), len(video_ids))
    for param, data in zip(params, results):
        curr_entropy = entropy(data)

        sim_matrix += torch.exp(-curr_entropy) * data

    return min_max_normalize(sim_matrix)


def fusion_reciprocal_rank(args, results, params, queries, video_ids, K=60):
    def get_ranks(data):
        ranked_data = torch.zeros_like(data)
        for i in range(data.size(0)):  # Loop over queries (rows)
            row = data[i]  # D [0.3, 0.5, 0.1,]
            sorted_indices = torch.argsort(row, descending=True)  # D [1, 0, 2]
            ranks = torch.zeros_like(sorted_indices)  # D [0, 0, 0]
            ranks[sorted_indices] = torch.arange(1, len(row) + 1)  # D [2, 1, 3]
            ranked_data[i] = ranks
        return ranked_data

    sim_matrix = torch.zeros(len(queries), len(video_ids))
    for param, data in zip(params, results):
        data = get_ranks(data)  # Q x D
        data = data.float()
        data = data + K
        data = 1 / data
        sim_matrix += data

    return sim_matrix

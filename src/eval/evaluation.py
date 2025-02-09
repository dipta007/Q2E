import torch
import numpy as np
from torchmetrics.retrieval import RetrievalRecall, RetrievalMRR, RetrievalNormalizedDCG, RetrievalPrecision, RetrievalMAP


def get_mean_median(sim_mat, targets):
    Q, V = sim_mat.shape
    assert sim_mat.shape == targets.shape, "Similarity matrix and targets must have same shape"

    # Get ranks for each query
    ranks = []

    for q in range(Q):
        # Get similarity scores for current query
        query_scores = sim_mat[q]

        # Get relevant video indices for current query
        relevant_indices = np.where(targets[q] == 1)[0]

        if len(relevant_indices) == 0:
            continue

        # Get ranks of all videos (argsort of negative scores to rank in descending order)
        video_ranks = np.argsort(-query_scores)

        # Find where the relevant videos appear in the ranking
        for rel_idx in relevant_indices:
            rank = np.where(video_ranks == rel_idx)[0][0] + 1  # +1 for 1-based ranking
            ranks.append(rank)

    ranks = np.array(ranks)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)

    return mean_rank, median_rank


def retrieval_score(sim_mat, targets):
    """
    targets: Q X V
    Q: number of queries
    V: number of videos
    """
    mean_median = get_mean_median(sim_mat, targets)

    targets = targets.flatten()

    # indices
    indices = [[i] * sim_mat.shape[1] for i in range(sim_mat.shape[0])]
    indices = torch.tensor(indices).flatten().long()

    # probabilities
    probs = sim_mat.flatten()
    probs = torch.tensor(probs)

    r1 = RetrievalRecall(top_k=1, aggregation="mean")
    r5 = RetrievalRecall(top_k=5, aggregation="mean")
    r10 = RetrievalRecall(top_k=10, aggregation="mean")

    pre1 = RetrievalPrecision(top_k=1, aggregation="mean")
    pre5 = RetrievalPrecision(top_k=5, aggregation="mean")
    pre10 = RetrievalPrecision(top_k=10, aggregation="mean")

    mrr = RetrievalMRR(top_k=10)
    ndcg = RetrievalNormalizedDCG(top_k=10)
    map = RetrievalMAP(top_k=10)

    metrics = {
        "R1": r1(probs, targets, indexes=indices).item() * 100,
        "R5": r5(probs, targets, indexes=indices).item() * 100,
        "R10": r10(probs, targets, indexes=indices).item() * 100,
        "Pre1": pre1(probs, targets, indexes=indices).item() * 100,
        "Pre5": pre5(probs, targets, indexes=indices).item() * 100,
        "Pre10": pre10(probs, targets, indexes=indices).item() * 100,
        "MRR": mrr(probs, targets, indexes=indices).item(),
        "NDCG": ndcg(probs, targets, indexes=indices).item() * 100,
        "MAP": map(probs, targets, indexes=indices).item() * 100,
        "MeanRank": mean_median[0],
        "MedianRank": mean_median[1],
    }
    return metrics

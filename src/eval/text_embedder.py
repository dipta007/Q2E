import warnings

import numpy as np
import torch
from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RAG = None
sbert = None


def get_one_to_one_score(args, seqs1, seqs2):
    """
    seqs1:  T1      - List of queries
    seqs2:  T2      - List of documents
    return: T1 x T2 - Similarity score matrix ranging from 0 to 100
    """

    def use_sentence_transformers():
        nonlocal seqs1, seqs2
        global sbert
        if sbert is None:
            sbert = SentenceTransformer(args.text_emb_model, device="cuda")

        seqs1_emb = sbert.encode(seqs1)  # T1 x D
        seqs2_emb = sbert.encode(seqs2)  # T2 x D

        # scores = util.dot_score(seqs1_emb, seqs2_emb) # T1 x T2
        scores = util.cos_sim(seqs1_emb, seqs2_emb)  # T1 x T2
        # convert score to [0, 1]
        scores = (scores + 1) / 2

        logit_scale = torch.tensor([100.0]).to(scores.device)
        scores = scores * logit_scale

        return scores

    def use_colbert():
        nonlocal seqs1, seqs2
        global RAG
        if RAG is None:
            RAG = RAGPretrainedModel.from_pretrained(args.text_emb_model, verbose=0, n_gpu=1)

        RAG.encode([x for x in seqs2], verbose=False, bsize=1024)
        results = RAG.search_encoded_docs(query=seqs1, k=len(seqs2), bsize=8192)

        scores = np.zeros((len(seqs1), len(seqs2)))
        result_indices = np.array([[res["result_index"] for res in row] for row in results])
        scores_values = np.array([[res["score"] for res in row] for row in results])
        row_indices = np.arange(len(results))[:, np.newaxis]
        scores[row_indices, result_indices] = scores_values

        RAG.clear_encoded_docs(force=True)

        return torch.tensor(scores)

    if args.text_emb_type == "colbert":
        scores = use_colbert()
    elif args.text_emb_type == "sentence-transformers":
        scores = use_sentence_transformers()

    return scores


def get_many_to_many_score(args, queries, docs):
    """
    queries:    T x X   - List of events
    docs:       V x Y   - List of clip captions
    return:     T x V   - Similarity score matrix ranging from 0 to 100
    """
    if isinstance(queries[0], str):
        queries = [[q] for q in queries]

    if isinstance(docs[0], str):
        docs = [[d] for d in docs]

    mx_queries = max([len(q) for q in queries])
    mx_docs = max([len(d) for d in docs])
    # todo: only for pyscene detect
    # mx_docs = min(mx_docs, 100)

    for i, query in enumerate(queries):
        queries[i] += (mx_queries - len(query)) * [""]

    for i, doc in enumerate(docs):
        docs[i] += (mx_docs - len(doc)) * [""]

    scores = torch.zeros(len(queries), len(docs))  # T x V

    with tqdm(total=mx_docs, desc="Computing many-to-many score", leave=True) as pbar:
        for j in range(mx_docs):
            curr_docs = [doc[j] for doc in docs]  # V
            cum_curr_queries = []
            for i in range(mx_queries):
                curr_queries = [query[i] for query in queries]  # T
                cum_curr_queries.extend(curr_queries)

            curr_cum_score = get_one_to_one_score(args, cum_curr_queries, curr_docs)  # (T * mx_queries) x V

            for i in range(0, len(curr_cum_score), len(queries)):
                curr_score = curr_cum_score[i : i + len(queries)]

                curr_queries = cum_curr_queries[i : i + len(queries)]
                for k, query in enumerate(curr_queries):
                    if not query or len(query) == 0:
                        curr_score[k, :] = float("-inf")
                for k, doc in enumerate(curr_docs):
                    if not doc or len(doc) == 0:
                        curr_score[:, k] = float("-inf")

                scores = torch.max(scores, curr_score)

            pbar.update(1)

    return scores


if __name__ == "__main__":
    pass

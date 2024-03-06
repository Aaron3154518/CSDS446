import numpy as np
import networkx as nx
import karateclub as kc
from sklearn.metrics import roc_auc_score
import math

import dcsbm
import latent_space_model as lsm


from itertools import combinations
from typing import Any, Callable, Dict, Iterator, Tuple

ScoreFunc = Callable[[int, int, nx.Graph], float]


def get_adj(graph: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(graph, weight=None)
    # Sometimes it zeroes the diagonals, sometimes not
    if A[0, 0] == 1:
        A -= np.identity(graph.number_of_nodes())
    return A


# Degree-Corrected Stochastic Block Model
def dcsbm_params(graph: nx.Graph) -> Dict[str, Any]:
    A = get_adj(graph)
    g = dcsbm.regularized_spectral_clustering(A, 2)
    _, P = dcsbm.parameter_estimation(A, g)
    return {"g": g, "P": P}


def dcsbm_score(
    n: int, m: int, graph: nx.Graph, g: np.ndarray = None, P: np.ndarray = None
) -> float:
    return P[g[n], g[m]]


# Latent Space Model
def lsm_params(graph: nx.Graph) -> Dict[str, Any]:
    A = get_adj(graph)
    X, a, L, res = lsm.estimateParams(A, dim=2)
    return {"X": X, "a": a}


def lsm_score(
    n: int, m: int, graph: nx.Graph, X: np.ndarray = None, a: float = 0
) -> float:
    return a - np.linalg.norm(X[n] - X[m])


# Deep Walk
def deep_walk_embeddings(graph: nx.Graph) -> Dict[str, Any]:
    deep_walk = kc.DeepWalk(dimensions=2)
    deep_walk.fit(graph)
    return {"x": deep_walk.get_embedding()}


def deep_walk_score(n: int, m: int, graph: nx.Graph, x: np.ndarray = None) -> float:
    return x[n] @ x[m]


def random(n: int, m: int, graph: nx.Graph) -> float:
    return np.random.uniform(0, 1)


def missing_links(graph: nx.Graph) -> Iterator[Tuple[int, int]]:
    return filter(lambda ns: not graph.has_edge(*ns), combinations(graph.nodes, 2))


def evaluate(
    graph: nx.Graph,
    score: ScoreFunc,
    true_links: np.ndarray,
    k: int = 8,
    T: int = 1,
    print_links: bool = False,
    kwargs_func: Callable[[nx.Graph], Dict[str, Any]] = lambda g: {},
):
    missing = np.array(list(missing_links(graph)))
    top_score = acc = auroc = 0
    links = None
    max_acc = 0
    for _ in range(T):
        kwargs = kwargs_func(graph)
        scores = np.array([(n, m, score(n, m, graph, **kwargs)) for n, m in missing])

        top_idxs = np.argsort(scores[:, 2])[::-1][:k]

        top_score += scores[top_idxs].sum()
        link_acc = true_links[top_idxs].sum()
        acc += link_acc
        auroc += roc_auc_score(true_links, scores[:, 2])

        if links is None or link_acc > max_acc:
            links = top_idxs
            max_acc = link_acc

    name = score.__name__
    if name.endswith("_score"):
        name = name[: -len("_score")]
    print(name)
    if T > 1:
        print("Averaged over", T, "trials")
    print("Top", k, "Score:", top_score / T)
    print("Accuracy:", acc / T, "/", k)
    print("AUROC:", auroc / T)
    if print_links:
        print(f"Top {k} Links for best trial:")
        print(
            ", ".join(
                ("*" * int(true_links[idx]))
                + f"{missing[idx, 0] + 1}->{missing[idx, 1] + 1}"
                for idx in links
            )
        )
    print()


if __name__ == "__main__":
    # Zero index karate club graph
    graph = nx.karate_club_graph()
    rm_links = [
        (i - 1, j - 1)
        for i, j in [
            (1, 5),
            (2, 4),
            (3, 29),
            (6, 17),
            (9, 34),
            (16, 33),
            (24, 26),
            (25, 32),
        ]
    ]
    graph.remove_edges_from(rm_links)
    true_links = np.array([(n, m) in rm_links for n, m in missing_links(graph)])

    score_funcs = [
        (dcsbm_params, dcsbm_score),
        (lsm_params, lsm_score),
        (deep_walk_embeddings, deep_walk_score),
    ]

    evaluate(graph, random, true_links, T=100)
    for params, score in score_funcs:
        evaluate(
            graph, score, true_links, k=10, T=5, print_links=True, kwargs_func=params
        )

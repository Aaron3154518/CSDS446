import math
from typing import Callable, Iterator, Tuple
from itertools import combinations

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score


# Score functions
ScoreFunc = Callable[[int, int, nx.Graph], float]


def preferential_attachment(n: int, m: int, graph: nx.Graph) -> float:
    return graph.degree(n) * graph.degree(m)


def common_neighbors(n: int, m: int, graph: nx.Graph) -> float:
    return len(set(graph.neighbors(n)) & set(graph.neighbors(m)))


def jacquard(n: int, m: int, graph: nx.Graph) -> float:
    n_n, n_m = set(graph.neighbors(n)), set(graph.neighbors(m))
    return len(n_n & n_m) / len(n_n | n_m)


def adamic_adar(n: int, m: int, graph: nx.Graph) -> float:
    neighbors = set(graph.neighbors(n)) & set(graph.neighbors(m))
    return sum(
        1 / math.log(graph.degree(k)) if graph.degree(k) > 1 else 0 for k in neighbors
    )


def random(n: int, m: int, graph: nx.Graph) -> float:
    return np.random.uniform(0, 1)


# Creates an iterator over missing edges
def missing_links(graph: nx.Graph) -> Iterator[Tuple[int, int]]:
    return filter(lambda ns: not graph.has_edge(*ns), combinations(graph.nodes, 2))


# Evaluates a scoring method for link prediction
def evaluate(
    graph: nx.Graph, score: ScoreFunc, true_links: np.ndarray, k: int = 8, T: int = 1
):
    top_score = acc = auroc = 0
    for _ in range(T):
        scores = np.array([(n, m, score(n, m, graph)) for n, m in missing_links(graph)])

        top_idxs = np.argpartition(scores[:, 2], -k)[-k:]

        top_score += scores[top_idxs].sum()
        acc += true_links[top_idxs].sum()
        auroc += roc_auc_score(true_links, scores[:, 2])

    print(score.__name__)
    if T > 1:
        print("Averaged over", T, "trials")
    print("Top", k, "Score:", top_score / T)
    print("Accuracy:", acc / T, "/", k)
    print("AUROC:", auroc / T)
    print()


def main():
    # Load the graph
    graph = nx.read_gml("karate.gml")

    # Remove the designated edges
    rm_links = [(1, 5), (2, 4), (3, 29), (6, 17), (9, 34), (16, 33), (24, 26), (25, 32)]
    graph.remove_edges_from(rm_links)

    # Create a binary array of true labels (1 if a missing link was removed from the graph)
    true_links = np.array([(n, m) in rm_links for n, m in missing_links(graph)])

    # Evaluate for each scoring method
    evaluate(graph, random, true_links, T=100)
    for score in [preferential_attachment, common_neighbors, jacquard, adamic_adar]:
        evaluate(graph, score, true_links)


if __name__ == "__main__":
    main()

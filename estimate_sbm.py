import numpy as np
from typing import List, Tuple
from math import log
from itertools import product


def plog(x: float) -> float:
    return log(x) if x != 0 else 0


def groups(g: np.ndarray) -> List[List[int]]:
    return [np.argwhere(g == i).flatten().tolist() for i in range(2)]


def calc_M(A: np.ndarray, groups: List[List[int]]) -> np.ndarray:
    return np.array(
        [[A[groups[r]][:, groups[s]].sum() for s in range(2)] for r in range(2)]
    )


def P_mle(A: np.ndarray, g: np.ndarray) -> np.ndarray:
    gs = groups(g)
    M = calc_M(A, gs)
    return np.array(
        [
            [
                M[r, s] / len(gs[r]) / len(gs[s]) if len(gs[s]) * len(gs[r]) != 0 else 0
                for s in range(2)
            ]
            for r in range(2)
        ]
    )


def Pg_mle(A: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    n = len(A)
    max_l = max_g = max_P = None
    for i in range(2**n):
        g = np.array([int(ch) for ch in f"{i:0<{n}b}"])
        gs = groups(g)
        M = calc_M(A, gs)
        P = P_mle(A, g)
        l = 0.5 * sum(
            M[r, s] * plog(P[r, s])
            + (len(gs[r]) * len(gs[s]) - M[r, s]) * plog(1 - P[r, s])
            for r, s in product(range(2), range(2))
        )
        if max_l is None or l > max_l:
            max_l, max_g, max_P = l, g, P

    return max_l, max_g, max_P


if __name__ == "__main__":
    A = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )
    g = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    print("MLE P:", P_mle(A, g))
    print()

    max_l, max_g, max_P = Pg_mle(A)

    print("Finding MLE Group Assignment:)
    print("Maximum Likelihood:", max_l)
    print("Group Assignment:", [[g_i + 1 for g_i in g] for g in groups(max_g)])
    print("P:", max_P)

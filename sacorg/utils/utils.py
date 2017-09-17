import time
import numpy as np
from math import factorial
from itertools import permutations as perm


def binomial(n, r):
    """
    Computes binomial coefficient (n,r)
    :param n:
    :param r:
    :return:
    """
    val = factorial(n) / (factorial(r) * factorial(n - r))
    return val


def vector_of_counts(deg_seq, dim=-1):
    """
    Compute the vector of counts for deg_seq
    :param deg_seq: degree sequence
    :param dim: length of the output sequence
    :return vector_of_counts: vector of counts for deg_seq
    """
    if dim < 0:
        dim = max(deg_seq)

    vector_of_counts = np.asarray([np.sum(deg_seq == value) for value in range(1, dim + 1)])
    return vector_of_counts


def conjugate(deg_seq, dim=-1):
    """
    Computes the conjugate of the given degree sequence deg_seq
    :param deg_seq: degree sequence
    :param dim: length of the output sequence
    :return conj_d: conjugate of deg_seq
    """
    if dim < 0:
        dim = max(deg_seq)

    if len(deg_seq) > 0:
        conj_d = np.asarray([np.sum(deg_seq >= num) for num in np.arange(1, dim + 1)])
    else:
        conj_d = np.zeros((1, dim), dtype=np.int)

    return conj_d


def matrix2edges(matrix, graph_type):
    """
    Converts a given binary matrix to the corresponding set of edges of an undirected graph
    :param matrix: Symmetric binary matrix having zero diagonal entries
    :return edges: Set of edges where vertex labels start from 1
    """
    # Get the row and column size
    n = matrix.shape[0]
    m = matrix.shape[1]
    # Ouput edge sequence
    edges = []
    if graph_type == "simple":
        for i in range(n):
            for j in range(i, n):
                if matrix[i, j] == 1:
                    edges.append([i + 1, j + 1])

    if graph_type == "bipartite":
        for i in range(n):
            for j in range(m):
                if matrix[i, j] == 1:
                    edges.append([i + 1, j + n + 1])

    return edges
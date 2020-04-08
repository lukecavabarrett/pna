import math
from queue import Queue

import numpy as np


def is_connected(A):
    """
    :param A:np.array the adjacency matrix
    :return:bool whether the graph is connected or not
    """
    for _ in range(int(1 + math.ceil(math.log2(A.shape[0])))):
        A = np.dot(A, A)
    return np.min(A) > 0


def identity(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return:F
    """
    return F


def first_neighbours(A):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the number of nodes reachable in 1 hop
    """
    return np.sum(A > 0, axis=0)


def second_neighbours(A):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the number of nodes reachable in no more than 2 hops
    """
    A = A > 0.0
    A = A + np.dot(A, A)
    np.fill_diagonal(A, 0)
    return np.sum(A > 0, axis=0)


def kth_neighbours(A, k):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the number of nodes reachable in k hops
    """
    A = A > 0.0
    R = np.zeros(A.shape)
    for _ in range(k):
        R = np.dot(R, A) + A
    np.fill_diagonal(R, 0)
    return np.sum(R > 0, axis=0)


def map_reduce_neighbourhood(A, F, f_reduce, f_map=None, hops=1, consider_itself=False):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, map its neighbourhood with f_map, and reduce it with f_reduce
    """
    if f_map is not None:
        F = f_map(F)
    A = np.array(A)

    A = A > 0
    R = np.zeros(A.shape)
    for _ in range(hops):
        R = np.dot(R, A) + A
    np.fill_diagonal(R, 1 if consider_itself else 0)
    R = R > 0

    return np.array([f_reduce(F[R[i]]) for i in range(A.shape[0])])


def max_neighbourhood(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the maximum in its neighbourhood
    """
    return map_reduce_neighbourhood(A, F, np.max, consider_itself=True)


def min_neighbourhood(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the minimum in its neighbourhood
    """
    return map_reduce_neighbourhood(A, F, np.min, consider_itself=True)


def std_neighbourhood(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the standard deviation of its neighbourhood
    """
    return map_reduce_neighbourhood(A, F, np.std, consider_itself=True)


def mean_neighbourhood(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the mean of its neighbourhood
    """
    return map_reduce_neighbourhood(A, F, np.mean, consider_itself=True)


def local_maxima(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, whether it is the maximum in its neighbourhood
    """
    return F == map_reduce_neighbourhood(A, F, np.max, consider_itself=True)


def graph_laplacian(A):
    """
    :param A:np.array the adjacency matrix
    :return: the laplacian of the adjacency matrix
    """
    L = (A > 0) * -1
    np.fill_diagonal(L, np.sum(A > 0, axis=0))
    return L


def graph_laplacian_features(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the laplacian of the adjacency matrix multiplied by the features
    """
    return np.matmul(graph_laplacian(A), F)


def isomorphism(A1, A2, F1=None, F2=None):
    """
        Takes two adjacency matrices (A1,A2) and (optionally) two lists of features. It uses Weisfeiler-Lehman algorithms, so false positives might arise
        :param      A1: adj_matrix, N*N numpy matrix
        :param      A2: adj_matrix, N*N numpy matrix
        :param      F1: node_values, numpy array of size N
        :param      F1: node_values, numpy array of size N
        :return:    isomorphic: boolean which is false when the two graphs are not isomorphic, true when they probably are.
    """
    N = A1.shape[0]
    if (F1 is None) ^ (F2 is None):
        raise ValueError("either both or none between F1,F2 must be defined.")
    if F1 is None:
        # Assign same initial value to each node
        F1 = np.ones(N, int)
        F2 = np.ones(N, int)
    else:
        if not np.array_equal(np.sort(F1), np.sort(F2)):
            return False
        if F1.dtype() != int:
            raise NotImplementedError('Still have to implement this')

    p = 1000000007

    def mapping(F):
        return (F * 234 + 133) % 1000000007

    def adjacency_hash(F):
        F = np.sort(F)
        b = 257

        h = 0
        for f in F:
            h = (b * h + f) % 1000000007
        return h

    for i in range(N):
        F1 = map_reduce_neighbourhood(A1, F1, adjacency_hash, f_map=mapping, consider_itself=True, hops=1)
        F2 = map_reduce_neighbourhood(A2, F2, adjacency_hash, f_map=mapping, consider_itself=True, hops=1)
        if not np.array_equal(np.sort(F1), np.sort(F2)):
            return False
    return True


def count_edges(A):
    """
    :param A:np.array the adjacency matrix
    :return: the number of edges in the graph
    """
    return np.sum(A) / 2


def is_eulerian_cyclable(A):
    """
    :param A:np.array the adjacency matrix
    :return: whether the graph has an eulerian cycle
    """
    return is_connected(A) and np.count_nonzero(first_neighbours(A) % 2 == 1) == 0


def is_eulerian_percorrible(A):
    """
    :param A:np.array the adjacency matrix
    :return: whether the graph has an eulerian path
    """
    return is_connected(A) and np.count_nonzero(first_neighbours(A) % 2 == 1) in [0, 2]


def map_reduce_graph(A, F, f_reduce):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the features of the nodes reduced by f_reduce
    """
    return f_reduce(F)


def mean_graph(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the mean of the features
    """
    return map_reduce_graph(A, F, np.mean)


def max_graph(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the maximum of the features
    """
    return map_reduce_graph(A, F, np.max)


def min_graph(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the minimum of the features
    """
    return map_reduce_graph(A, F, np.min)


def std_graph(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: the standard deviation of the features
    """
    return map_reduce_graph(A, F, np.std)


def has_hamiltonian_cycle(A):
    """
    :param A:np.array the adjacency matrix
    :return:bool whether the graph has an hamiltonian cycle
    """
    A += np.transpose(A)
    A = A > 0
    V = A.shape[0]

    def ham_cycle_loop(pos):
        if pos == V:
            if A[path[pos - 1]][path[0]]:
                return True
            else:
                return False
        for v in range(1, V):
            if A[path[pos - 1]][v] and not used[v]:
                path[pos] = v
                used[v] = True
                if ham_cycle_loop(pos + 1):
                    return True
                path[pos] = -1
                used[v] = False
        return False

    used = [False] * V
    path = [-1] * V
    path[0] = 0

    return ham_cycle_loop(1)


def all_pairs_shortest_paths(A, inf_sub=math.inf):
    """
    :param A:np.array the adjacency matrix
    :param inf_sub: the placeholder value to use for pairs which are not connected
    :return:np.array all pairs shortest paths
    """
    A = np.array(A)
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if A[i][j] == 0:
                A[i][j] = math.inf
            if i == j:
                A[i][j] = 0

    for k in range(N):
        for i in range(N):
            for j in range(N):
                A[i][j] = min(A[i][j], A[i][k] + A[k][j])

    A = np.where(A == math.inf, inf_sub, A)
    return A


def diameter(A):
    """
    :param A:np.array the adjacency matrix
    :return: the diameter of the gra[h
    """
    sum = np.sum(A)
    apsp = all_pairs_shortest_paths(A)
    apsp = np.where(apsp < sum + 1, apsp, -1)
    return np.max(apsp)


def eccentricity(A):
    """
    :param A:np.array the adjacency matrix
    :return: the eccentricity of the gra[h
    """
    sum = np.sum(A)
    apsp = all_pairs_shortest_paths(A)
    apsp = np.where(apsp < sum + 1, apsp, -1)
    return np.max(apsp, axis=0)


def sssp_predecessor(A, F):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array the nodes features
    :return: for each node, the best next step to reach the designated source
    """
    assert (np.sum(F) == 1)
    assert (np.max(F) == 1)
    s = np.argmax(F)
    N = A.shape[0]
    P = np.zeros(A.shape)
    V = np.zeros(N)
    bfs = Queue()
    bfs.put(s)
    V[s] = 1
    while not bfs.empty():
        u = bfs.get()
        for v in range(N):
            if A[u][v] > 0 and V[v] == 0:
                V[v] = 1
                P[v][u] = 1
                bfs.put(v)
    return P


def max_eigenvalue(A):
    """
    :param A:np.array the adjacency matrix
    :return: the maximum eigenvalue of A
    since A is positive symmetric, all the eigenvalues are guaranteed to be real
    """
    [W, _] = np.linalg.eig(A)
    return W[np.argmax(np.absolute(W))].real


def max_eigenvalues(A, k):
    """
    :param A:np.array the adjacency matrix
    :param k:int the number of eigenvalues to be selected
    :return: the k greatest (by absolute value) eigenvalues of A
    """
    [W, _] = np.linalg.eig(A)
    values = W[sorted(range(len(W)), key=lambda x: -np.absolute(W[x]))[:k]]
    return values.real


def max_absolute_eigenvalues(A, k):
    """
    :param A:np.array the adjacency matrix
    :param k:int the number of eigenvalues to be selected
    :return: the absolute value of the k greatest (by absolute value) eigenvalues of A
    """
    return np.absolute(max_eigenvalues(A, k))


def max_absolute_eigenvalues_laplacian(A, n):
    """
    :param A:np.array the adjacency matrix
    :param k:int the number of eigenvalues to be selected
    :return: the absolute value of the k greatest (by absolute value) eigenvalues of the laplacian of A
    """
    A = graph_laplacian(A)
    return np.absolute(max_eigenvalues(A, n))


def max_eigenvector(A):
    """
    :param A:np.array the adjacency matrix
    :return: the maximum (by absolute value) eigenvector of A
    since A is positive symmetric, all the eigenvectors are guaranteed to be real
    """
    [W, V] = np.linalg.eig(A)
    return V[:, np.argmax(np.absolute(W))].real


def spectral_radius(A):
    """
    :param A:np.array the adjacency matrix
    :return: the maximum (by absolute value) eigenvector of A
    since A is positive symmetric, all the eigenvectors are guaranteed to be real
    """
    return np.abs(max_eigenvalue(A))


def page_rank(A, F=None, iter=64):
    """
    :param A:np.array the adjacency matrix
    :param F:np.array with initial weights. If None, uniform initialization will happen.
    :param iter: log2 of length of power iteration
    :return: for each node, its pagerank
    """

    # normalize A rows
    A = np.array(A)
    A /= A.sum(axis=1)[:, np.newaxis]

    # power iteration
    for _ in range(iter):
        A = np.matmul(A, A)

    # generate prior distribution
    if F is None:
        F = np.ones(A.shape[-1])
    else:
        F = np.array(F)

    # normalize prior
    F /= np.sum(F)

    # compute limit distribution
    return np.matmul(F, A)


def tsp_length(A, F=None):
    """
        :param A:np.array the adjacency matrix
        :param F:np.array determining which nodes are to be visited. If None, all of them are.
        :return: the length of the Traveling Salesman Problem shortest solution
    """

    A = all_pairs_shortest_paths(A)
    N = A.shape[0]
    if F is None:
        F = np.ones(N)
    targets = np.nonzero(F)[0]
    T = targets.shape[0]
    S = (1 << T)
    dp = np.zeros((S, T))

    def popcount(x):
        b = 0
        while x > 0:
            x &= x - 1
            b += 1
        return b

    msks = np.argsort(np.vectorize(popcount)(np.arange(S)))
    for i in range(T + 1):
        for j in range(T):
            if (1 << j) & msks[i] == 0:
                dp[msks[i]][j] = math.inf

    for i in range(T + 1, S):
        msk = msks[i]
        for u in range(T):
            if (1 << u) & msk == 0:
                dp[msk][u] = math.inf
                continue
            cost = math.inf
            for v in range(T):
                if v == u or (1 << v) & msk == 0:
                    continue
                cost = min(cost, dp[msk ^ (1 << u)][v] + A[targets[v]][targets[u]])
            dp[msk][u] = cost
    return np.min(dp[S - 1])


def get_nodes_labels(A, F):
    """
    Takes the adjacency matrix and the list of nodes features (and a list of algorithms) and returns
    a set of labels for each node
    :param      A: adj_matrix, N*N numpy matrix
    :param      F: node_values, numpy array of size N
    :return:    labels: KxN numpy matrix where K is the number of labels for each node
    """
    labels = [identity(A, F), map_reduce_neighbourhood(A, F, np.mean, consider_itself=True),
              map_reduce_neighbourhood(A, F, np.max, consider_itself=True),
              map_reduce_neighbourhood(A, F, np.std, consider_itself=True), first_neighbours(A), second_neighbours(A),
              eccentricity(A)]
    return np.swapaxes(np.stack(labels), 0, 1)


def get_graph_labels(A, F):
    """
    Takes the adjacency matrix and the list of nodes features (and a list of algorithms) and returns
    a set of labels for the whole graph
    :param      A: adj_matrix, N*N numpy matrix
    :param      F: node_values, numpy array of size N
    :return:    labels: numpy array of size K where K is the number of labels for the graph
    """
    labels = [diameter(A)]
    return np.asarray(labels)

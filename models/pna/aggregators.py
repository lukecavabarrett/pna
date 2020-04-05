import math
import torch
EPS = 1e-5

# each aggregator is a function taking as input X (B x N x N x Din), adj (B x N x N), self_loop and device and
# returning the aggregated value of X (B x N x Din) for each dimension

def aggregate_identity(X, adj, self_loop=False, device='cpu'):
    # Y is corresponds to the elements of the main diagonal of X
    (_, N, N, _) = X.shape
    Y = torch.sum(torch.mul(X, torch.eye(N).reshape(1, N, N, 1)), dim=2)
    return Y


def aggregate_mean(X, adj, self_loop=False, device='cpu'):
    # D^{-1} A * X    i.e. the mean of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    D = torch.sum(adj, -1, keepdim=True)
    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    X_mean = torch.div(X_sum, D)
    return X_mean


def aggregate_max(X, adj, min_value=-math.inf, self_loop=False, device='cpu'):
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    adj = adj.unsqueeze(-1)  # adding extra dimension
    M = torch.where(adj > 0.0, X, torch.tensor(min_value, device=device))
    max = torch.max(M, -3)[0]
    return max


def aggregate_min(X, adj, max_value=math.inf, self_loop=False, device='cpu'):
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    adj = adj.unsqueeze(-1)  # adding extra dimension
    M = torch.where(adj > 0.0, X, torch.tensor(max_value, device=device))
    min = torch.min(M, -3)[0]
    return min


def aggregate_std(X, adj, self_loop=False, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X, adj, self_loop, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_var(X, adj, self_loop=False, device='cpu'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    D = torch.sum(adj, -1, keepdim=True)
    X_sum_squares = torch.sum(torch.mul(torch.mul(X, X), adj.unsqueeze(-1)), dim=2)
    X_mean_squares = torch.div(X_sum_squares, D)  # D^{-1} A X^2
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_mean_squares - torch.mul(X_mean, X_mean))  # relu(mean_squares_X - mean_X^2)
    return var


def aggregate_sum(X, adj, self_loop=False, device='cpu'):
    # A * X    i.e. the mean of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    return X_sum


def aggregate_normalised_mean(X, adj, self_loop=False, device='cpu'):
    # D^{-1/2] A D^{-1/2] X
    (B, N, N, _) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device)
                   .unsqueeze(0).repeat(B, 1, 1))  # D^{-1/2]
    adj = torch.matmul(torch.matmul(rD, adj), rD)  # D^{-1/2] A' D^{-1/2]

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    return X_sum


def aggregate_softmax(X, adj, self_loop=False, device='cpu'):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_exp = torch.exp(X)
    adj = adj.unsqueeze(-1)  # adding extra dimension
    X_exp = torch.mul(X_exp, adj)
    X_sum = torch.sum(X_exp, dim=2, keepdim=True)
    softmax = torch.sum(torch.mul(torch.div(X_exp, X_sum), X), dim=2)
    return softmax


def aggregate_softmin(X, adj, self_loop=False, device='cpu'):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, self_loop=self_loop, device=device)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'identity': aggregate_identity, 'std': aggregate_std, 'var': aggregate_var,
               'normalised_mean': aggregate_normalised_mean,
               'softmax': aggregate_softmax, 'softmin': aggregate_softmin}

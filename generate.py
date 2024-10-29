# imports
import numpy as np
import torch

# generate forwards weights
def generate_fwd_weights(n_vpn, n_kc, k, q):
    # t based on k
    t = int(2 * n_kc * k / n_vpn)
    n_vpn_half = int(n_vpn / 2)
    
    # Initialize connection weights for the combined MBs
    weights = torch.zeros(2 * n_kc, n_vpn)
    
    # Define q matrices for sampling
    q_matrix_left = torch.tensor([[q] * n_kc + [1 - q] * n_kc] * n_vpn, dtype=torch.float).T
    q_matrix_right = torch.tensor([[1 - q] * n_kc + [q] * n_kc] * n_vpn, dtype=torch.float).T
    
    # Randomly choose connected indices for each VPN with skew for combined MBs
    for i in range(n_vpn_half):
        connected_indices_left = torch.multinomial(q_matrix_left[:, i], t, replacement=False)
        weights[connected_indices_left, i] = 1 / k

    for i in range(n_vpn_half, n_vpn):
        connected_indices_right = torch.multinomial(q_matrix_right[:, i], t, replacement=False)
        weights[connected_indices_right, i] = 1 / k
    
    # Split weights into left and right MBs
    weights_left = weights[:n_kc, :]
    weights_right = weights[n_kc:, :]

    # Return weights as sparse matrices
    return weights_left.to_sparse(), weights_right.to_sparse()

# generate random and forwards weights
def generate_rnd_weights(n_vpn, n_kc, k, q):
    # t based on k
    t = int(2 * n_kc * k / n_vpn)
    n_vpn_half = int(n_vpn / 2)
    
    # Initialize connection weights for the combined MBs
    weights = torch.zeros(2 * n_kc, n_vpn)
    
    # Define q matrices for sampling
    q_matrix_left = torch.tensor([[q] * n_kc + [1 - q] * n_kc] * n_vpn, dtype=torch.float).T
    q_matrix_right = torch.tensor([[1 - q] * n_kc + [q] * n_kc] * n_vpn, dtype=torch.float).T

    # random weights matrix
    random_weights = 2*torch.rand((2*n_kc, n_vpn))/k
    
    # Randomly choose connected indices for each VPN with skew for combined MBs
    for i in range(n_vpn_half):
        connected_indices_left = torch.multinomial(q_matrix_left[:, i], t, replacement=False)
        weights[connected_indices_left, i] = random_weights[connected_indices_left, i]

    for i in range(n_vpn_half, n_vpn):
        connected_indices_right = torch.multinomial(q_matrix_right[:, i], t, replacement=False)
        weights[connected_indices_right, i] = random_weights[connected_indices_right, i]
    
    # Split weights into left and right MBs
    weights_left = weights[:n_kc, :]
    weights_right = weights[n_kc:, :]

    # Return weights as sparse matrices
    return weights_left.to_sparse(), weights_right.to_sparse()
import numpy as np
from scipy import linalg
import torch


def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x joints_num(22) x 3
    pred_joints: num_poses x joints_num(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x joints_num=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_cuda_compatible(matrix1, matrix2):
    """
    CUDA-friendly distance matrix calculation.
    Uses torch.cdist which is optimized for GPU.
    """
    return torch.cdist(matrix1, matrix2, p=2)


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat
    
def calculate_R_precision_cuda_compatible(embedding1, embedding2, top_k, sum_all=False):
    """
    Calculates R-precision on GPU.
    """
    # Get N x N distance matrix
    dist_mat = euclidean_distance_matrix_cuda_compatible(embedding1, embedding2)
    
    # Get indices of top_k smallest distances (closest matches)
    # largest=False because smaller distance means higher similarity
    _, topk_indices = torch.topk(dist_mat, k=top_k, dim=1, largest=False)
    
    # The correct match for row i is index i
    gt_indices = torch.arange(embedding1.shape[0], device=embedding1.device).view(-1, 1)
    
    # Check if gt_index exists in the top_k indices for each row
    top_k_mat = (topk_indices == gt_indices)
    
    # To mimic your original calculate_top_k cumulative logic:
    # We check if 'any' of the columns up to k are true
    cumulative_hits = []
    for i in range(1, top_k + 1):
        hit_at_k = top_k_mat[:, :i].any(dim=1, keepdim=True)
        cumulative_hits.append(hit_at_k)
    
    result = torch.cat(cumulative_hits, dim=1).float()
    
    return result.sum(dim=0) if sum_all else result


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist

def calculate_matching_score_cuda_compatible(embedding1, embedding2, sum_all=False):
    # Vectorized L2 norm on GPU
    dist = torch.linalg.norm(embedding1 - embedding2, dim=1)
    return dist.sum() if sum_all else dist

def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_activation_statistics_cuda_compatible(activations):
    """
    activations: (N, D) tensor
    """
    mu = torch.mean(activations, dim=0)
    # PyTorch cov expects variables as rows, observations as columns, or use correction
    cov = torch.cov(activations.T) 
    return mu, cov

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_diversity_cuda_compatible(activation, diversity_times):
    num_samples = activation.shape[0]
    # Random selection on GPU
    idx1 = torch.randperm(num_samples, device=activation.device)[:diversity_times]
    idx2 = torch.randperm(num_samples, device=activation.device)[:diversity_times]
    
    dist = torch.linalg.norm(activation[idx1] - activation[idx2], dim=1)
    return dist.mean()

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def calculate_multimodality_cuda_compatible(activation, multimodality_times):
    # activation: (N, num_per_sent, D)
    num_per_sent = activation.shape[1]
    
    idx1 = torch.randint(0, num_per_sent, (multimodality_times,), device=activation.device)
    idx2 = torch.randint(0, num_per_sent, (multimodality_times,), device=activation.device)
    
    dist = torch.linalg.norm(activation[:, idx1] - activation[:, idx2], dim=2)
    return dist.mean()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_frechet_distance_cuda_compatible(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Robust PyTorch-native implementation of Frechet Distance.
    Works on GPU and handles numerical instability.
    """
    # Perform math in float64 for high precision, then convert back
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    diff = mu1 - mu2

    # 1. Compute the square root of sigma1
    # Since sigma1 is a covariance matrix, it is symmetric positive semi-definite
    vals1, vecs1 = torch.linalg.eigh(sigma1)
    # Clamp eigenvalues to 0 to prevent negatives from noise
    sqrt_sigma1 = vecs1 @ torch.diag(torch.sqrt(torch.relu(vals1))) @ vecs1.T

    # 2. Compute the matrix for the middle term: sqrt(sigma1) * sigma2 * sqrt(sigma1)
    # This matrix is also symmetric, which is the key to stability
    middle_matrix = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    vals2, _ = torch.linalg.eigh(middle_matrix)
    
    # 3. Tr(sqrt(sigma1 * sigma2)) is equivalent to Tr(sqrt(sqrt_sigma1 * sigma2 * sqrt_sigma1))
    # This is a much more stable identity
    tr_covmean = torch.sum(torch.sqrt(torch.relu(vals2)))

    # 4. Final FID calculation
    fid = (diff @ diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    
    # Force result to be at least 0 (it can only be negative due to precision error)
    return max(0.0, fid.item())


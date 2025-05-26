import numpy as np

def compute_best_fit_transform(points, targets, weights=None):
    """
    Computes the best-fit transform T (R, t) that maps points -> targets
    using optional weights.
    Returns 4x4 transformation matrix.
    """
    if weights is None:
        weights = np.ones(len(points))

    weights = weights / np.sum(weights)

    # Weighted centroids
    p_centroid = np.average(points, axis=0, weights=weights)
    t_centroid = np.average(targets, axis=0, weights=weights)

    # Centered
    p_centered = points - p_centroid
    t_centered = targets - t_centroid

    # Weighted covariance
    W = np.diag(weights)
    H = p_centered.T @ W @ t_centered

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = t_centroid - R @ p_centroid

    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

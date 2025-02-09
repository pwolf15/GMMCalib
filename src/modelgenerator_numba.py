import open3d as o3d
import numpy as np
import numba
from numba import jit, prange

@jit(nopython=True, parallel=True)
def sse(A, B):
    """Compute Sum of Squared Errors (SSE) ensuring shape consistency."""
    if A.ndim == 1:
        A = A.reshape(1, -1)  # Ensure (1,3) if it's a single point
    if B.ndim == 1:
        B = B.reshape(1, -1)  # Ensure (1,3) if it's a single point

    N, D1 = A.shape
    K, D2 = B.shape

    assert D1 == D2, f"Shape mismatch: A.shape={A.shape}, B.shape={B.shape}"

    # Expand dimensions for proper broadcasting
    A_exp = A[:, np.newaxis, :]  # (N, 1, D)
    B_exp = B[np.newaxis, :, :]  # (1, K, D)

    # Compute SSE across dimensions
    diff = A_exp - B_exp
    return np.sum(diff ** 2, axis=2)  # Sum over dimension D (XYZ)



@jit(nopython=True, parallel=True)
def compute_alpha(TV, X, pk, Q, beta):
    """Compute posteriors (alpha) in parallel."""
    num_views = len(TV)
    K = X.shape[1]
    alpha = np.empty((num_views, K))

    for i in prange(num_views):
        alpha[i, :] = np.exp(-0.5 * np.transpose(Q) * sse(TV[i], X))

    alpha = alpha.T / (np.sum(alpha, axis=1) + beta)
    return alpha.T

@jit(nopython=True, parallel=True)
def update_transformations(V, X, alpha, Q):
    """Compute optimal rotation and translation matrices in parallel."""
    num_views = len(V)
    R = np.empty((num_views, 3, 3))
    t = np.empty((num_views, 3))

    for i in prange(num_views):
        lmda = np.sum(alpha[i], axis=0).T
        W = np.multiply(np.dot(V[i], alpha[i]), Q.T)
        b = np.multiply(lmda, Q)

        mW = np.sum(W, axis=1)
        mX = np.dot(X, b)
        sumOfWeights = np.sum(lmda.T.dot(Q))

        P = np.dot(X, W.T) - (np.dot(mX, mW.T) / sumOfWeights)

        # Compute SVD for rotation
        u, _, v = np.linalg.svd(P)
        R[i] = np.dot(u, np.dot(np.diag([1, 1, np.linalg.det(np.dot(u, v.T))]), v))
        t[i] = (mX - np.dot(R[i], mW)) / sumOfWeights

    return R, t

def jgmm_numba(V, Xin, maxNumIter):
    """Fast implementation of Joint GMM Model using Numba for acceleration."""
    V = [np.transpose(i) for i in V]
    X = np.transpose(Xin)
    M = len(V)  # Number of measurements
    dim, K = X.shape  # Number of centroids

    R = [np.eye(3) for _ in range(M)]  # Rotation matrices
    t = [np.zeros(3) for _ in range(M)]  # Translation vectors
    TV = [np.dot(R[i], V[i]) + t[i].reshape((3, 1)) for i in range(M)]  # Transformed sets

    # Initialize Covariances
    minXYZ = np.min([np.min(TVX, axis=1) for TVX in TV] + [np.min(X, axis=1)], axis=0).reshape((dim, 1))
    maxXYZ = np.max([np.max(TVX, axis=1) for TVX in TV] + [np.max(X, axis=1)], axis=0).reshape((dim, 1))
    Q = (1 / sse(minXYZ, maxXYZ)).reshape((K, 1)).astype(np.float64)

    epsilon, gamma = 1e-9, 0.1
    pk = 1 / (K * (gamma + 1))
    beta = gamma / ((2 / np.mean(Q)) * (gamma + 1))

    for it in range(maxNumIter):
        print(f"GMM Iteration: {it}")

        # Compute posteriors
        alpha = compute_alpha(TV, X, pk, Q, beta)

        # Compute new transformations
        R, t = update_transformations(V, X, alpha, Q)

        # Apply transformations
        TV = [np.dot(R[i], V[i]) + t[i].reshape((3, 1)) for i in range(M)]

        # Update GMM centroids
        lmdaMatrix = np.asarray([np.sum(alpha[i], axis=0).T for i in range(M)]).astype(np.float64)
        den = np.sum(lmdaMatrix, axis=0).T
        X = np.sum(np.stack([TV[i].dot(alpha[i]) for i in range(M)], axis=0), axis=0) / den

        # Update Covariances
        wnormes = [np.sum(np.multiply(alpha[i], sse(TV[i].astype(np.float64), X)), axis=0) for i in range(M)]
        Q = np.transpose(3 * den / (np.sum(np.stack(wnormes, axis=0), axis=0) + 3 * den * epsilon))

        # Update Priors
        pk = den / ((gamma + 1) * sum(den))

    return X, TV, (R, t), pk

import arrayfire as af
import numpy as np
import open3d as o3d
from arrayfire import to_array

def af_sse(A, B):
    """Compute the sum of squared errors using ArrayFire (GPU acceleration)."""
    A_af = af.to_array(A.astype(np.float32))  # Ensure float32
    B_af = af.to_array(B.astype(np.float32))
    diff = A_af - af.tile(B_af, (A_af.dims()[0], 1))
    return af.sum(diff ** 2, dim=0)

def jgmm_af(V, Xin, maxNumIter):
    """Joint GMM Model optimized using ArrayFire for GPU acceleration."""
    V = [af.to_array(np.transpose(i)) for i in V]
    X = af.to_array(np.transpose(Xin))

    M = len(V)  # Number of measurements
    dim, K = X.dims()  # Number of centroids

    # Initialize rotation & translation matrices
    R = [af.identity(3, 3) for _ in range(M)]
    t = [af.constant(0, 3) for _ in range(M)]
    
    TV = []
    for i in range(M):
        # Debugging Shapes
        print(f"Shape of R[{i}]:", R[i].dims())
        print(f"Shape of V[{i}]:", V[i].dims())
        print(f"Shape of t[{i}]:", t[i].dims())

        # Ensure correct transformation
        t_reshaped = af.reshape(t[i], 3, 1)
        t_tiled = af.tile(t_reshaped, (1, V[i].dims()[1]))  # Match the number of points

        TV.append(af.matmul(R[i], V[i]) + t_tiled)
        
    # Initialize Covariances
    minXYZ = af.min(af.join(1, *[af.min(TVX, 1) for TVX in TV], af.min(X, 1)), 1)
    maxXYZ = af.max(af.join(1, *[af.max(TVX, 1) for TVX in TV], af.max(X, 1)), 1)

    Q = af.reciprocal(af_sse(minXYZ, maxXYZ)).reshape(K, 1)

    epsilon, gamma = 1e-9, 0.1
    pk = 1 / (K * (gamma + 1))
    beta = gamma / ((2 / af.mean(Q)) * (gamma + 1))

    for it in range(maxNumIter):
        print(f"GMM Iteration: {it}")

        # Compute posteriors (alpha)
        alpha = [af.exp(-0.5 * af.transpose(Q) * af_sse(TV[i], X)) for i in range(M)]
        alpha = [af.transpose(a) / (af.sum(a, dim=1) + beta) for a in alpha]

        # Compute transformations
        lmda = [af.sum(a, dim=0).T for a in alpha]
        W = [af.matmul(V[i], alpha[i]) * af.tile(Q.T, (V[i].dims()[1], 1)) for i in range(M)]
        b = [lmda[i] * Q for i in range(M)]

        mW = [af.sum(W[i], dim=1) for i in range(M)]
        mX = [af.matmul(X, b[i]) for i in range(M)]
        sumOfWeights = [af.sum(lmda[i].T * Q) for i in range(M)]

        P = [af.matmul(X, W[i].T) - (af.matmul(mX[i], mW[i].T) / sumOfWeights[i]) for i in range(M)]

        # Compute SVD for rotation
        R, t = [], []
        for i in range(M):
            U, S, Vt = af.svd(P[i])

            # Fix determinant issue
            det_val = af.det(af.matmul(U, Vt.T))
            diag_matrix = af.diag(af.to_array([1, 1, det_val]))

            Ri = af.matmul(U, af.matmul(diag_matrix, Vt))
            ti = (mX[i] - af.matmul(Ri, mW[i])) / sumOfWeights[i]

            R.append(Ri)
            t.append(ti)

        # Apply transformations
        TV = [af.matmul(R[i], V[i]) + t[i].reshape(3, 1) for i in range(M)]

        # Update GMM centroids
        lmdaMatrix = af.stack([af.sum(alpha[i], dim=0).T for i in range(M)], dim=1)
        den = af.sum(lmdaMatrix, dim=1).T
        X = af.sum(af.stack([af.matmul(TV[i], alpha[i]) for i in range(M)], dim=0), dim=0) / den

        # Update Covariances
        wnormes = [af.sum(af_sse(TV[i].T.astype(af.Dtype.f64), X) * alpha[i], dim=0) for i in range(M)]
        Q = 3 * den / (af.sum(af.stack(wnormes, dim=0), dim=0) + 3 * den * epsilon)

        # Update Priors
        pk = den / ((gamma + 1) * af.sum(den))

    return X.to_ndarray(), [tv.to_ndarray() for tv in TV], ([r.to_ndarray() for r in R], [t.to_ndarray() for t in t]), pk.to_ndarray()

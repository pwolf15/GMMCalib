import arrayfire as af
import numpy as np
import open3d as o3d

import arrayfire as af
import numpy as np
import open3d as o3d

def af_sse(A, B):
    """Compute the sum of squared errors using ArrayFire (GPU acceleration)."""
    A_af = af.array(A)
    B_af = af.array(B)
    diff = A_af - af.tile(B_af, (A_af.dims()[0], 1))
    return af.sum(diff ** 2, dim=0)

def jgmm_af(V, Xin, maxNumIter):
    """Joint GMM Model optimized using ArrayFire for GPU acceleration."""
    V = [af.array(np.transpose(i)) for i in V]
    X = af.array(np.transpose(Xin))
    M = len(V)  # Number of measurements
    dim, K = X.dims()  # Number of centroids

    # Initialize rotation & translation matrices
    R = [af.identity(3, 3) for _ in range(M)]
    t = [af.constant(0, 3) for _ in range(M)]
    
    TV = [af.matmul(R[i], V[i]) + t[i].reshape(3, 1) for i in range(M)]

    # Initialize Covariances
    minXYZ = af.min(af.stack([af.min(TVX, dim=1) for TVX in TV] + [af.min(X, dim=1)], dim=1), dim=1)
    maxXYZ = af.max(af.stack([af.max(TVX, dim=1) for TVX in TV] + [af.max(X, dim=1)], dim=1), dim=1)
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
            Ri = af.matmul(U, af.matmul(af.diag([1, 1, af.det(af.matmul(U, Vt.T))]), Vt))
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

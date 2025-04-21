import numpy as np

def print_stats(name, arr):
    """Helper to print stats of a numpy array"""
    print(f"{name}: shape={arr.shape}, mean={np.mean(arr):.6f}, std={np.std(arr):.6f}, min={np.min(arr):.6f}, max={np.max(arr):.6f}")

def sqe(Y, X):

    """
    Compute squared Euclidean distance between each row in Y and X
    Y: (N,3), X: (K, 3)
    Returns: (N, K)
    """
    diff = Y[:, np.newaxis, :] - X[np.newaxis, :, :] # (N,K,3)
    return np.sum(diff**2, axis=2) # (N,K)

def jrmpc(V, X, max_num_iter=100, gamma=0.1, R=None, T=None, Q=None, epsilon=None, update_priors=None):

    # JRMPC: Joint Registration of Multiple Point Clouds
    # [R,T] = JRMPC(V,X) estimates Euclidean transformation parameters R,t
    #   in order to rigidly register the views in V.
    # V is an Mx1 cell array of views, with each view V{j} j = 1:M
    #   represented as a 3xN matrix of (cartesian) coordinates.
    #   A column of V{j}(:,i) of an element V{j} in V contains 
    #   coordinates of [x y z]^T of i-th 3d point of j-th view
    # X is a 3 x K matrix of initial centers.

    ### CHECKS

    M = len(V)
    print(f'Num point clouds: {M}')

    # GMM params
    K, dim = X.shape
    print(f'GMM dim: {dim}, K: {K}')

    print('X shape', X.shape)
    if dim != 3:
        print('X must be a 3xK matrix')
        exit(1)
    
    for j in range(0, M):
        if V[j].shape[1] != 3:
            print('V must be an M-length list of 3x .. matrices')
            print(f'V[{j}] has {V[j].shape[1]} in dimension 1.')

    ### END CHECKS

    ### optional params
    is_set_R = 0
    is_set_T = 0
    is_set_Q = 0
    is_set_max_num_iter = 0
    is_set_initial_priors_or_gamma = 0
    is_set_epsilon = 0
    is_set_update_priors = 0

    
    ### initialize defaults
    if not is_set_R:
        R = [np.eye(3) for _ in range(M)]

    if not is_set_T:
        print('view shape: ', V[0].shape, 'X shape: ', X.shape)
        t = [(np.mean(X, axis=0) - np.mean(view, axis=0)).reshape(3, 1) for view in V]

    TV = [(view @ Rv.T + tv.T) for view, Rv, tv in zip(V, R, t)]

    # Q represents weights: inverse variance
    # initial spread (uncertainty of Gaussians)
    # larger space, less precision
    if not is_set_Q:
        all_sets = TV + [X]
        print(all_sets[0].shape)
        min_xyz = [np.min(pcd, axis=0) for pcd in all_sets]
        max_xyz = [np.max(pcd, axis=0) for pcd in all_sets]

        min_xyz = np.min(np.stack(min_xyz), axis=0)
        max_xyz = np.max(np.stack(max_xyz), axis=0)
        d_squared = np.sum((min_xyz - max_xyz)**2)
        Q = np.full(K, 1.0/d_squared)
        print(np.mean(Q))

    if not is_set_epsilon:
        epsilon = 1e-6
    if not is_set_update_priors:
        update_priors = 0
    if not is_set_initial_priors_or_gamma:
        gamma = 1/K
        pk = np.full(K, 1/(K+1))
    gamma = 0.1
    pk = np.full(K, 1/(K*(gamma+1)))
    print('pk', pk.shape)

    # parameter h in the paper (this should be proportional to the volume that
    # encompasses all the point sets). Above, we partially translate the sets around 
    # (0,0,0), and we compute accordingly the initial varainces (and precisions)
    # Thus, we compute h in a similar way.

    h = 2 / np.mean(Q)
    beta = gamma /(h * (gamma + 1))
    print('h,beta', h,beta)
    # exit(1)

    Q_col = Q.reshape(K, 1)  # for b and dot products
    Q_row = Q.reshape(1, K)  # for a and W

    pk = pk.reshape(1, -1)

    for i in range(0, max_num_iter):

        # posteriors

        # sqe (squared differences between TV and X)
        a = [sqe(view, X) for view in TV]

        # compute responsibilities
        # pk * S^-1.5*exp(-.5/S^2*||.||)
        # remember a = responsibilities
        # represents how likely a given component k is responsible for point n in view i
        a = [pk * (Q_row ** 1.5) * np.exp(-0.5 * Q_row * a_i) for a_i in a]
        # print_stats('responsibilities', a[0])

        # normalize
        a = [a_i / (np.sum(a_i, axis=1, keepdims=True) + beta) for a_i in a]
        # print_stats('normalized', a[0])

        # weighted (Umeyama)/ procrustes M-step
        # goal: find best rigid transformation that aligns GMM-weighted version
        # of observed point cloud to current estimate of GMM centers (X), under 
        # soft assignments (a)

        # total responsibility per Gaussian, per view
        lambda_ = [np.sum(a_i, axis=0).reshape(-1, 1) for a_i in a]
        print_stats('lambda', lambda_[0])

        # weighted means scaled by precision
        W = [(a_i.T @ v) * Q_col for v, a_i in zip(V, a)]
        # print_stats('v', V[0])
        # print_stats('W', W[0])

        # weights, b
        b = [np.multiply(lmbd, Q_col) for lmbd in lambda_]
        print_stats('b', b[0])

        # mean of W
        mW = [np.sum(W_i, axis=0, keepdims=True) for W_i in W]
        # print_stats('mW', mW[0])

        # mean of X 
        mX = [b_i.T @ X for b_i in b]
        # print_stats('mX', mX[0])

        sum_of_weights = [np.dot(lmbd.T, Q_col)[0, 0] for lmbd in lambda_]
        # print_stats('sum_of_weights', sum_of_weights[0])

        # P cross-covariance matrix
        # print('X shape', X.shape)
        # print('W.shape', W[0].shape)
        # print('mX shape', mX[0].shape)
        # print('mW shape', mW[0].shape)

        # print_stats("P0 term1", term1)
        # print_stats("P0 term2", term2)
        # print_stats("P0", P0)

        P = [X.T @ W_i - mX_i.T @ mW_i / s 
                for W_i, s, mW_i, mX_i in zip(W, sum_of_weights, mW, mX)]

        # print_stats('X', X)
        # print(X[0][0])
        # print_stats('b', b[0])
        # print('W[0]', W[0][0][0])
        # print_stats('W', W[0])
        # print_stats('mW', mW[0])
        # print('mW[0]', mW[0][0][0])
        # print_stats('mX', mX[0])
        # print('mX[0]', mX[0][0][0])
        # print_stats('sum_of_weights', sum_of_weights[0])
        # print_stats('P', P[0])
        uu = []
        vv = []

        # compute SVD for each covariance matrix
        for P_i in P:
            U, _, Vt = np.linalg.svd(P_i)
            uu.append(U)
            vv.append(Vt.T)

        R = []
        # rotation with reflection check
        for u, v in zip(uu, vv):
            det_uv = np.linalg.det(u @ v.T)
            D = np.diag([1, 1, det_uv])
            R.append(u @ D @ v.T)

        # translation vectors
        t = [(mX_i - mW_i @ R_i.T) / s for mX_i, mW_i, R_i, s in zip(mX, mW, R, sum_of_weights)]

        # print_stats("mW[0]", mW[0])  
        # print_stats("mX[0]", mX[0])  
        # print_stats("R[0]", R[0])  
        # print_stats("sumOfWeights[0]", sum_of_weights[0])  
        # print_stats("t[0]", t[0])


        # next e-step
        # apply transformation to each TV
        # print('shapes', V[0].shape, R[0].shape, t[0].shape)
        TV = [view @ R_i.T + t_i for view, R_i, t_i in zip(V, R, t)]
        # print_stats("tv[0]", TV[0])
        # print(TV[0].shape)

        # update GMM centers
        den = np.sum(np.hstack(lambda_), axis=1)
        # print_stats('den', den)

        # compute each view contribution to GMM centers
        # for i, (tv, a_i) in enumerate(zip(TV, a)):
        #     print(f"[{i}] tv.T shape: {tv.T.shape}, a_i shape: {a_i.shape}")

        X = [a_i.T @ tv for tv, a_i in zip(TV, a)]
        X_sum = np.sum(np.stack(X, axis=0), axis=0)
        X = X_sum / den[:, np.newaxis]
        print_stats('X', X)


        # update precision / variance
        wnormes = [np.sum(a_i * sqe(tv, X), axis=0) for tv, a_i in zip(TV, a)]
        wnorm_sum = np.sum(np.stack(wnormes, axis=0), axis=0)
        Q = (3 * den) / (wnorm_sum + 3 * den * epsilon)
        Q_col = Q.reshape(K, 1)
        Q_row = Q.reshape(1, K)


        print(f"[iter {i}] X mean: {np.mean(X):.5f}, Q mean: {np.mean(Q):.5f}")

        # print_stats("wnormes", wnormes[0])
        # print_stats("Q", Q)
        if update_priors:
            pk = den / ((gamma + 1) * np.sum(den))
            pk = pk.reshape(1, -1)
    return R, t, X
import open3d as o3d
import numpy as np

import jax.numpy as jnp
from jax import jit, vmap

import time
###################################################################################
#########           M O D E L    G E N E R A T I O N     ##########################
###################################################################################

def get_registrations(V, nObs, num_sensors):

    initial_positions = {}
    initial_positions["sensors"] = list(range(1, num_sensors + 1))
    initial_positions["num_obs"] = nObs // num_sensors
    initial_positions["pcd_list"] = []
    for idx in range(0, nObs):
        sensor_id = idx // initial_positions["num_obs"] 
        obs_id = idx % initial_positions["num_obs"] 
        initial_positions["pcd_list"].append({
            "file": f'{sensor_id}_{obs_id}',
            "points": V[idx].T.tolist()
        })
    return initial_positions

"""
This code implementation was inpired by G. D. Evangelidis and R. Horaud, 
“Joint Alignment of Multiple Point Sets with Batch and Incremental Expectation-Maximization,” 
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 1397–1410, June 2018.
"""

def jgmm(V, Xin, maxNumIter, socket_client=None, fixCentroids=False, num_sensors=2):
    """Calculate the transformations and jointly align points clouds
    Parameters
    ---------------
    V: list, M
        containing the measurement point clouds
    Xin: (dim, K) np.array
        initial GMM centroids as Point Cloud
    Returns
    ---------------
    X: array, shape (N, 3)
        jgmm model point cloud
    TV: list, shape (M)
        Transformed views
    T: list,
        Transformations at each iteration
    pk: array, shape(K, 1)
        probability of each point in the generated model

    """

    # -------------------------------------------------------------------- #
    #               Initialize Variables and Matrices                      #
    # -------------------------------------------------------------------- #

    V = [np.transpose(i) for i in V]
    X = np.transpose(Xin)
    TV = [] 

    """Number of Measurments"""
    M = len(V)
    """Number of Centroids """
    dim, K = X.shape

    """Init rotation matrix"""
    R = []
    for i in range(M):
        R.append(np.eye(3))

    """ Init translation matrix"""
    t = []
    for i in range(len(V)):
        t.append(np.array([0, 0, 0]))

    """ Transformed Sets based on initial R & t"""
    TV = [np.dot(R[i],V[i]) + t[i].reshape((3,1)) for i in range(len(V))]

    """ Initial Covariances for the centroids"""
    minXYZ, maxXYZ = [], []
    TVX = TV.copy()
    TVX.append(X)
    for i in range(len(TVX)):
        minXYZ.append(np.min(TVX[i], axis = 1))
        maxXYZ.append(np.max(TVX[i], axis = 1))

    minXYZ = np.min(minXYZ, axis=0).reshape((dim,1))
    maxXYZ = np.max(maxXYZ, axis=0).reshape((dim,1))

    Q = np.multiply(np.ones((1, K)), (1/ sse_jax(minXYZ, maxXYZ) ) ).reshape((K,1)).astype(np.float64)

    #maxNumIter = 10
    epsilon = 1e-9
    updatePriors = 1
    gamma =  0.1
    pk = 1/(K*(gamma+1))

    print(f"Fix centroids {fixCentroids}")
    # -------------------------------------------------------------------- #
    #               E    M    A L G O R I T H M                            #
    # -------------------------------------------------------------------- #

    h = np.divide(2, np.mean(Q))
    beta = np.divide(gamma, np.multiply(h, gamma+1))
    pk = np.transpose(pk)
    T = []

    for it in range(maxNumIter):
        print("GMM Iteration: ", it)
        ''' Calculate Posteriors '''
        ''' Squared Error Between transformed frames and compontents'''
        alpha = [sse_jax(jnp.asarray(i), X) for i in TV]
        ''' Correspondences '''
        alpha = [
            pk * (Q.T ** 1.5) * jnp.exp(-0.5 * Q.T * i)
            for i in alpha
        ]

        '''Normalization with the sum of alpha and beta'''
        alpha = [
            i / (jnp.sum(i, axis=1, keepdims=True) + beta)
            for i in alpha
        ]

        print("alpha[0] shape:", alpha[0].shape)

        '''Weights '''
        lmda = [np.sum(i, axis= 0).T  for i in alpha]

        W = [np.multiply(np.dot(V[i], alpha[i]), Q.T) for i in range(len(V))]

        b = [np.multiply(i, Q) for i in lmda]

        '''mean of W'''
        mW = [np.sum(i, axis= 1) for i in W]

        '''mean of X'''
        mX = [np.dot(X, i) for i in b]

        sumOfWeights = [i.T.dot(Q)[0,0] for i in lmda]

        P = [np.dot(X, W[i].T)- (np.dot(mX[i], mW[i].T)/sumOfWeights[i])  for i in range(len(W))]

        '''SVD'''
        uu, ss, vv = [], [],[]
        for i in range(len(P)):
            u, s, v = np.linalg.svd(P[i])
            uu.append(u)
            ss.append(s)
            vv.append(v)


        '''Find optimal rotation'''

        R = [np.dot( uu[i].dot(np.diag([1, 1, np.linalg.det(np.dot(uu[i], vv[i].T))])),vv[i]) for i in range(len(uu))]

        ''' Find optimal translation'''
        t = [(mX[i] - R[i].dot(mW[i])) / sumOfWeights[i] for i in range(len(R))]


        '''Populate T'''
        T.append((R, t))

        '''Transformed Sets'''
        TV = [R[i].dot(V[i]) + t[i] for i in range(len(R))]

        '''Update X'''
        lmdaMatrix = np.asarray(lmda).astype(np.float64)
        den = np.sum(np.moveaxis(lmdaMatrix, 0, 1), axis=1).T

        if not fixCentroids:
            X = [TV[i].dot(alpha[i]) for i in range(len(TV))] # (M, 3, K) Matrix
            X = np.sum(np.stack(np.asarray(X[:]), axis=0), axis=0)
            X = X/den
            
        if socket_client:
            # send gmm means
            print('TV', len(TV))
            socket_client.emit("gmm_means", {"Xin": Xin.tolist(), "X": X.T.tolist(), "num_iter": it+1})
            registrations = get_registrations(TV, len(TV), num_sensors)
            registrations["num_iter"] = it+1
            socket_client.emit("registrations", registrations)

        '''Update Covariances '''
        wnormes = [np.sum(np.multiply(alpha[i], sse_jax(np.asarray(TV[i].astype(np.float64)), np.asarray(X))), axis=0) for i in range(len(TV))]

        Q = np.transpose(np.divide(3*den, np.sum(np.stack(np.asarray(wnormes[:]), axis=0), axis=0) + 3*den*epsilon))

        if updatePriors:
            pk = den / ((gamma+1)*sum(den))
 
    Q = np.divide(1, Q)
    return X, TV, T, pk


def sse(A, B):
    """Compute the Sum Of Squared Error of two matrices.
        Returns
        -------
        C : list, shape (N_train, 4)
            SSE of the two matrices.
        """
    start = time.time()
    A = np.moveaxis(A[np.newaxis, :, :], 0, -1) # results in a (3, N, 1) matrix
    B = np.swapaxes(np.moveaxis(B[np.newaxis, :, :], 0, -1), 1, -1) # results in a (3, 1, K) matrix

    C = np.sum(np.power((A - B), 2), axis=0) # sum over the the first axis of the A and B (three dimensions)
    end = time.time()

    print(f"JAX Time: {end - start:.4f} seconds")
    if isinstance(C, (list, tuple, np.ndarray)):
        return C
    else:
        return C[0][0]

from jax import jit

@jit
def sse_jax(A, B):
    """
    Compute the Sum of Squared Error between two matrices A and B.
    A: shape (3, N)
    B: shape (3, K)
    Returns: array of shape (N, K)
    """
    start = time.time()
    A_exp = jnp.expand_dims(A, axis=-1)  # shape (3, N, 1)
    B_exp = jnp.expand_dims(B, axis=1)   # shape (3, 1, K)
    diff = A_exp - B_exp                 # shape (3, N, K)
    end = time.time()

    print(f"JAX Time: {end - start:.4f} seconds")
    return jnp.sum(diff ** 2, axis=0)    # shape (N, K)


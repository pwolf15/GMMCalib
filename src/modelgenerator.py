import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

from scipy.spatial import cKDTree

def save_centroid_frames(X_history, output_folder="centroid_frames"):
    """
    Saves each iteration of the centroids as a PNG file.

    Parameters:
        X_history (list of np.array): List of centroid positions at each iteration.
        output_folder (str): Folder to save the images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, X in enumerate(X_history):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[0, :], X[1, :], X[2, :], c='blue', marker='o', label=f'Iteration {i}')

        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Centroids at Iteration {i}')
        ax.legend()

        filename = os.path.join(output_folder, f"frame_{i:03d}.png")
        plt.savefig(filename)
        plt.close(fig)  # Close figure to free memory


def create_gif_from_frames(output_folder="centroid_frames", gif_filename="centroid_evolution.gif", duration=0.5):
    """
    Creates a GIF from saved centroid PNG images.

    Parameters:
        output_folder (str): Folder containing the saved PNG frames.
        gif_filename (str): Output GIF filename.
        duration (float): Duration per frame in seconds.
    """
    images = []
    filenames = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
    
    for filename in filenames:
        file_path = os.path.join(output_folder, filename)
        images.append(imageio.imread(file_path))
    
    gif_path = os.path.join(output_folder, gif_filename)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved at: {gif_path}")


def project_onto_cube(X, cube_model, weight=1.0):
    """Project GMM centroids onto the cube surface using nearest neighbors."""
    cube_tree = cKDTree(cube_model.T)
    distances, indices = cube_tree.query(X.T)

    # Correct the shape to match X
    closest_points = cube_model[:, indices.flatten()]  # Flatten to avoid shape mismatch

    # Move X towards the closest points on the cube
    X_updated = X - weight * (X - closest_points)
    print(f'Before: {X[0][0]}, after: {X_updated[0][0]}')

    return X_updated

###################################################################################
#########           M O D E L    G E N E R A T I O N     ##########################
###################################################################################

"""
This code implementation was inpired by G. D. Evangelidis and R. Horaud, 
“Joint Alignment of Multiple Point Sets with Batch and Incremental Expectation-Maximization,” 
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 1397–1410, June 2018.
"""
def jgmm(V, Xin, maxNumIter):
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

    # store prior for comparison later
    X_initial = np.copy(X) 
    TV = [] 

    """Number of Measurments"""
    M = len(V)
    """Number of Centroids """
    dim, K = X.shape
    print("num centroids: ", K)

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

    Q = np.multiply(np.ones((1, K)), (1/ sse(minXYZ, maxXYZ) ) ).reshape((K,1)).astype(np.float64)

    #maxNumIter = 10
    epsilon = 1e-9
    updatePriors = 1
    gamma =  0.1
    pk = 1/(K*(gamma+1))


    # -------------------------------------------------------------------- #
    #               E    M    A L G O R I T H M                            #
    # -------------------------------------------------------------------- #

    h = np.divide(2, np.mean(Q))
    beta = np.divide(gamma, np.multiply(h, gamma+1))
    pk = np.transpose(pk)
    T = []

    X_history = []

    for it in range(maxNumIter):
        
        print("GMM Iteration: ", it)
        ''' Calculate Posteriors '''
        ''' Squared Error Between transformed frames and compontents'''
        alpha = [sse(np.asarray(i), np.asarray(X)) for i in TV]
        ''' Correspondences '''
        alpha = [np.multiply(np.multiply(pk, np.transpose(Q) ** 1.5),np.exp(np.multiply(-0.5 * np.transpose(Q), i))) for i in alpha]
        '''Normalization with the sum of alpha and beta'''
        alpha = [np.divide(i.T, np.asmatrix(np.sum(i, axis=1) + beta)).T for i in alpha]

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

        # Update GMM centroids
        X = [TV[i].dot(alpha[i]) for i in range(len(TV))] # (M, 3, K) Matrix
        X = np.sum(np.stack(np.asarray(X[:]), axis=0), axis=0)
        X = X/den

        # use projection
        # X_new = X - lambda * (X - X_closest points)
        # lambda_cube = weights
        # lambda_cube = 1.0
        # X_w_geo_prior = project_onto_cube(X, X_initial, weight=lambda_cube)
        # print(X.shape, X_w_geo_prior.shape)
        # X = X_w_geo_prior
        X_history.append(X.copy())

        '''Update Covariances '''
        wnormes = [np.sum(np.multiply(alpha[i], sse(np.asarray(TV[i].astype(np.float64)), np.asarray(X))), axis=0) for i in range(len(TV))]

        Q = np.transpose(np.divide(3*den, np.sum(np.stack(np.asarray(wnormes[:]), axis=0), axis=0) + 3*den*epsilon))

        if updatePriors:
            pk = den / ((gamma+1)*sum(den))

    save_centroid_frames(X_history)
    create_gif_from_frames()

    Q = np.divide(1, Q)
    return X, TV, T, pk


def sse(A, B):
    """Compute the Sum Of Squared Error of two matrices.
        Returns
        -------
        C : list, shape (N_train, 4)
            SSE of the two matrices.
        """

    A = np.moveaxis(A[np.newaxis, :, :], 0, -1) # results in a (3, N, 1) matrix
    B = np.swapaxes(np.moveaxis(B[np.newaxis, :, :], 0, -1), 1, -1) # results in a (3, 1, K) matrix

    C = np.sum(np.power((A - B), 2), axis=0) # sum over the the first axis of the A and B (three dimensions)
    if isinstance(C, (list, tuple, np.ndarray)):
        return C
    else:
        return C[0][0]
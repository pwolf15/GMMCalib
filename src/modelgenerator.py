import open3d as o3d
import numpy as np
###################################################################################
#########           M O D E L    G E N E R A T I O N     ##########################
###################################################################################

import pyceres
import numpy as np

def angle_axis_rotate_point(rot, point):
    """
    Rotate a point using an angle-axis vector.

    Parameters:
    - rot: 3D angle-axis rotation vector
    - point: 3D point to rotate

    Returns:
    - Rotated 3D point
    """
    theta = np.linalg.norm(rot)
    if theta < 1e-8:
        return point.copy()

    axis = rot / theta
    axis = axis.reshape(3)
    point = point.reshape(3)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    dot = np.dot(axis, point)
    cross = np.cross(axis, point)

    return (cos_theta * point +
            (1 - cos_theta) * dot * axis +
            sin_theta * cross)

# Residual function for pyceres
class LidarResidual(pyceres.CostFunction):
    def __init__(self, observed_point, latent_point, weight):
        super().__init__()
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([6])

        self.observed_point = np.asarray(observed_point, dtype=np.float64).reshape(3,)
        self.latent_point = np.asarray(latent_point, dtype=np.float64).reshape(3,)
        self.weight = float(np.squeeze(weight))

    def Evaluate(self, parameters, residuals, jacobians):
        extrinsics = parameters[0]  # shape (6,)
        rot_vec = extrinsics[:3]
        t_vec = extrinsics[3:]

        theta = np.linalg.norm(rot_vec)
        if theta < 1e-8:
            R = np.eye(3)
            dR_drot = np.zeros((3, 3, 3))  # dR/d(rot) is zero
        else:
            axis = rot_vec / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = (
                np.eye(3) * np.cos(theta)
                + (1 - np.cos(theta)) * np.outer(axis, axis)
                + np.sin(theta) * K
            )

            # Approximate Jacobian dR/d(rot_vec) using finite difference
            dR_drot = np.zeros((3, 3, 3))
            eps = 1e-6
            dR_drot = np.zeros((3, 3))  # residual_dim x rot_param_dim
            eps = 1e-6
            for j in range(3):  # for each rotation parameter
                d_rot = np.zeros(3)
                d_rot[j] = eps
                p_plus = angle_axis_rotate_point(rot_vec + d_rot, self.observed_point)
                p_minus = angle_axis_rotate_point(rot_vec - d_rot, self.observed_point)
                dp = (p_plus - p_minus) / (2 * eps)
                dR_drot[:, j] = dp

        p_transformed = angle_axis_rotate_point(rot_vec, self.observed_point) + t_vec
        residual = self.weight * (self.latent_point - p_transformed)
        for i in range(3):
            residuals[i] = residual[i]

        if jacobians is not None and jacobians[0] is not None:
            J = jacobians[0].reshape(3, 6)

            # Rotation Jacobian (finite difference)
            eps = 1e-6
            for j in range(3):  # for each rotation parameter
                d_rot = np.zeros(3)
                d_rot[j] = eps
                p_plus = angle_axis_rotate_point(rot_vec + d_rot, self.observed_point)
                p_minus = angle_axis_rotate_point(rot_vec - d_rot, self.observed_point)
                dp = (p_plus - p_minus) / (2 * eps)
                for i in range(3):
                    J[i, j] = -self.weight * dp[i]

            # Translation Jacobian: identity scaled
            J[:, 3:6] = -self.weight * np.eye(3)

        return True

# Bundle Adjustment integration
def bundle_adjustment(V, X, alpha, num_sensors, num_obs, initial_R, initial_t):
    problem = pyceres.Problem()

    import cv2

    extrinsics = []

    for R_i, t_i in zip(initial_R, initial_t):
        
        # convert rotations to angle-axis vectors
        rot_vec, _ = cv2.Rodrigues(R_i)        # shape (3,1)
        rot_vec = rot_vec.flatten()            # shape (3,)

        # flatten translation vectors
        t_vec = np.array(t_i).reshape(-1)      # guarantees shape (3,)

        assert rot_vec.shape == (3,), f"rot_vec shape: {rot_vec.shape}"
        assert t_vec.shape == (3,), f"t_vec shape: {t_vec.shape}"

        # create 6D pose vector per observation
        extrinsics.append(np.hstack([rot_vec, t_vec]))

    extrinsics_flat = np.array(extrinsics).flatten()

    print(len(alpha))
    print(alpha[0].shape)
    print(V[0].shape)
    
    for obs_idx, (V_obs, alpha_obs) in enumerate(zip(V, alpha)):
        for pt_idx in range(V_obs.shape[1]):  # each point
            weights = np.asarray(alpha[obs_idx][pt_idx, :]).flatten()
            observed = V_obs[:, pt_idx]  # shape (3,)
            for k, w in enumerate(weights):
                if w < 1e-4:
                    continue  # skip weak associations
                latent = X[:, k]  # GMM centroid
                normalized_weight = w / (np.max(weights) + 1e-8)

                k = np.argmax(weights)  # strongest GMM match
                if weights[k] < 1e-4:  # optionally, filter weak matches
                    continue
                latent = X[:, k]
                cost_fn = LidarResidual(observed, latent, normalized_weight)
                loss = pyceres.CauchyLoss(1.0)
                problem.add_residual_block(cost_fn, loss, [extrinsics[obs_idx]])  # NOT double-nested


    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.minimizer_progress_to_stdout = True

    # print("Initial cost estimate:", problem.evaluate(extrinsics)[0])

    summary = pyceres.SolverSummary()
    print("Before BA:", extrinsics[0])
    pyceres.solve(options, problem, summary)

    print(summary.BriefReport())
    print("After BA: ", extrinsics[0])

    num_observations = len(initial_R) 
    optimized_R = []
    optimized_t = []
    optimized_extrinsics = np.array(extrinsics)
    for i in range(num_observations):
        rot_vec = optimized_extrinsics[i, :3]
        t_vec = optimized_extrinsics[i, 3:]
        R_i, _ = cv2.Rodrigues(rot_vec)
        optimized_R.append(R_i)
        optimized_t.append(t_vec)

    return optimized_R, optimized_t

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

import plotly.graph_objects as go
import numpy as np

def plot_registered_observations(registration_data, num_iter, output_filename="registered_observations.png"):
    fig = go.Figure()

    sensors = registration_data["sensors"]
    num_obs = registration_data["num_obs"]
    pcd_list = registration_data["pcd_list"]

    sensor_colors = {
        0: "red",
        1: "blue", 
        2: "green", 
        3: "orange", 
        4: "purple", 
        5: "cyan"
    }

    for idx, pcd in enumerate(pcd_list):
        sensor_id = idx // num_obs
        obs_id = idx % num_obs
        pts = np.array(pcd["points"])
        color = sensor_colors.get(sensor_id, 'gray')
        label = f"Obs {obs_id + 1} - Sensor {sensor_id + 1}"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=2, color=color),
            name=label
        ))

    fig.update_layout(
        title=f"Registration after {num_iter} iterations",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        showlegend=True
    )

    fig.write_image(output_filename, width=1200, height=800)
    print(f"Saved registered point cloud image to: {output_filename}")

import plotly.graph_objects as go
import numpy as np
import os

def plot_registered_observations_multiview(registration_data, output_dir="output", basename="registered_final"):
    os.makedirs(output_dir, exist_ok=True)

    sensors = registration_data["sensors"]
    num_obs = registration_data["num_obs"]
    pcd_list = registration_data["pcd_list"]

    sensor_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "orange",
        4: "purple",
        5: "cyan"
    }

    views = {
        "top":     dict(eye=dict(x=0.0, y=0.0, z=2.5)),
        "side":    dict(eye=dict(x=2.5, y=0.0, z=0.0)),
        "front":   dict(eye=dict(x=0.0, y=2.5, z=0.0)),
        "iso":     dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    }

    for view_name, camera in views.items():
        fig = go.Figure()

        for idx, pcd in enumerate(pcd_list):
            sensor_id = idx // num_obs
            obs_id = idx % num_obs
            pts = np.array(pcd["points"])
            color = sensor_colors.get(sensor_id, 'gray')
            label = f"Obs {obs_id + 1} - Sensor {sensor_id + 1}"

            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=label
            ))

        fig.update_layout(
            title=f"Final Registration - {view_name.capitalize()} View",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
                camera=camera
            ),
            showlegend=False
        )

        output_path = os.path.join(output_dir, f"{basename}_{view_name}.png")
        fig.write_image(output_path, width=1200, height=800)
        print(f"Saved {view_name} view to: {output_path}")


"""
This code implementation was inpired by G. D. Evangelidis and R. Horaud, 
“Joint Alignment of Multiple Point Sets with Batch and Incremental Expectation-Maximization,” 
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 1397–1410, June 2018.
"""

def jgmm(V, Xin, maxNumIter, socket_client=None, fixCentroids=False, num_sensors=2, save_images=False):
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
    t = [(np.mean(Xin, axis=0) - np.mean(view.T, axis=0)) for view in V]
    # for i in range(len(V)):
    #     t.append(np.array([0, 0, 0]))

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
    gamma =  0.3
    pk = 1/(K*(gamma+1))

    print(f"Fix centroids {fixCentroids}")
    # -------------------------------------------------------------------- #
    #               E    M    A L G O R I T H M                            #
    # -------------------------------------------------------------------- #

    h = np.divide(2, np.mean(Q))
    beta = np.divide(gamma, np.multiply(h, gamma+1))
    pk = np.transpose(pk)
    T = []
    alpha = []

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
                    
        if save_images: 
            registrations = get_registrations(TV, len(TV), num_sensors)
            registrations["num_iter"] = it+1
            plot_registered_observations(registrations, it+1, f"output/registered_observations_{str(it+1)}.png")
    
        '''Update Covariances '''
        wnormes = [np.sum(np.multiply(alpha[i], sse(np.asarray(TV[i].astype(np.float64)), np.asarray(X))), axis=0) for i in range(len(TV))]

        Q = np.transpose(np.divide(3*den, np.sum(np.stack(np.asarray(wnormes[:]), axis=0), axis=0) + 3*den*epsilon))

        if updatePriors:
            pk = den / ((gamma+1)*sum(den))
 
    assert len(R) == M, f"Expected {M} sensor extrinsics, got {len(R)}"
    assert len(t) == M


    Q = np.divide(1, Q)
    print('T shape, len(T)', len(T))
    
    new_T = []

    use_ba = False
    if use_ba:
        optimized_R, optimized_t = bundle_adjustment(TV, X, alpha, num_sensors, len(TV)//2, R, t)
 
        for i in range(len(T)):
            t_list = []
            R_list = []
            for j in range(len(optimized_R)):
                R_list.append(optimized_R[j])
                t_list.append(optimized_t[j])
            new_T.append((R_list, t_list))
    else:
        for i in range(len(T)):
            t_list = []
            R_list = []
            for j in range(len(R)):
                R_list.append(R[j])
                t_list.append(t[j])
            new_T.append((R_list, t_list))
            
    print('T shape, len(T)', len(T))

    return X, TV, new_T, pk


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
import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R
import generatePCDs
import os
import math
from jrmpc import jrmpc

# example of using jrmpc into synthetic data
#   loads len(theta) views from ./syntheticData and calls
#   jrmpc to do the registration. It creates 4 plots:
#       1. initial position of point sets
#       2. registration at every iteration
#       3. final alignment achieved after maxNumIter
#       4. "cleaned up" point sets registration
#   ./syntheticData contains 4 partial views from stanford bunny
#   degrade each view iwht disparity noise and outliers
#   angles in theta are ground truth angles (same for all 3 axes)


def generate_random_transform():
    """Generate a random rotation matrix and translation vector."""
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(0, np.pi, 3))
    t = np.random.uniform(-0.5, 0.5, size=(3,))
    return R, t

def apply_transfrom(pcd, R, t):
    """Apply transformation to point cloud."""
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.rotate(R)
    pcd_copy.translate(t)
    return pcd_copy

def create_synthetic_views(base_pcd, M=5):
    """Create M transfomed versions of the base point cloud."""
    views = []
    transforms = []
    for _ in range(M):
        R, t = generate_random_transform()
        transformed = apply_transfrom(base_pcd, R, t)
        views.append(transformed)
        transforms.append((R,t))
    return views, transforms

def main1():
    # Load base point cloud
    base_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_poisson_disk(500)
    base_pcd.paint_uniform_color([1,0,0])

    # Create synthetic transformed views
    M = 5
    views, transforms = create_synthetic_views(base_pcd, M)

    # Visualize all views
    o3d.visualization.draw_geometries(views, window_name="Synthetic JRMPC")

def angle2rotation(theta):
    # apply rotation around X, Y, then Z with same angle
    rot = R.from_euler('zyx', [theta, theta, theta])
    return rot.as_matrix()

def main():

    # number of iterations to be run
    max_num_iter = 100

    # latent angles
    # latent angle: rotation angle used to generate the synthetic data
    # this angle is not given to the alignment algorithm
    # latent = not directly observed
    # rotating V{j} by theta(j) reprojects it to V{1} rotated by 
    # theta(1). used to quantify the accuracy of estimated R
    # jrmpc will tell you how close estimated rotations are to true rotations
    theta = [0, np.pi/20, np.pi/10, np.pi/6]

    # number of view, M files must be found in directory ./syntheticData
    M = len(theta)

    # load views, file view<j>.txt corresponds to theta(j)
    # cutView*.txt is a partial view as described in paper, while view*.txt
    # sees the whole surface (again downsampled and noisy
    config_file_path = "./config/bunny_config.yaml"
    data_path = "./bunny_data"
    sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    pcds, num_sensors = generatePCDs.generate_data(data_path, config_file_path, sequence)
    V = [np.array(cloud.points) for cloud in pcds]

    # initialize GMM means Xin, using random sampling of a unit sphere.
    # Choose your own initialization. You may want to initialize Xin with
    # some of the sets

    # set K as 50% of median cardinality of the views
    n_points = [view.shape[0] for view in V]
    K = math.ceil(0.5 * np.median(n_points))
    print(f'K: {K}')

    # sample the unit sphere, by randomly selecting azimuth / elevation
    # angles
    az = 2 * np.pi * np.random.rand(K)
    el = 2 * np.pi * np.random.rand(K)

    # convert spherical to Cartesian coordinates
    x = np.cos(az) * np.cos(el)
    y = np.sin(el)
    z = np.sin(az) * np.cos(el)
    Xin = np.vstack((x, y, z)) / 10

    gamma = 0.1
    R, t, X = jrmpc(V, Xin.T, max_num_iter, gamma)
    Rgt = []
    for i in range(0, M):
        Rgt.append(angle2rotation(theta[i]))

    for j in range(0, M):
        rel_est = R[j].T @ R[0]   # relative rotation from view j to view 0
        rel_gt = Rgt[j]  # true relative rotation from view j to view 0
        error_j = np.linalg.norm(rel_gt - rel_est, ord='fro')
        print('t: ', t[j])
        print(f'View {j}: error = {error_j:.6f}')

    # show initial position of point clouds

    # color using same coloring as MATLAB
    clrmap = [
        [1.0, 0.1412, 0.0],
        [0.1373, 0.4196, 0.5569],
        [0.0, 0.0, 1.0],
        [0.8039, 0.6078, 0.1137],
        [0.0, 0.0, 0.0], # Xin
    ]
    Xin_pcd = o3d.geometry.PointCloud()
    Xin_pcd.points = o3d.utility.Vector3dVector(Xin.T)
    pcds.append(Xin_pcd)
    for idx in range(0, len(pcds)):
        colors = np.tile(clrmap[idx], (np.asarray(pcds[idx].points).shape[0], 1))
        pcds[idx].colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(pcds, window_name="Initial position of the point clouds")


if __name__ == "__main__":
    main()
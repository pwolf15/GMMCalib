import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
from socket_client import SocketIOClient
import create_gt
import numpy as np
import pickle
import csv
from scipy.spatial.transform import Rotation as R
import math
import open3d as o3d
import yaml

from deformation import cad_deform_to_gmm
from deformable_model import DeformableShapeModel

import numpy as np
from scipy.spatial.transform import Rotation as R

import plotly.graph_objects as go

def to_xyz_array(pcd):
    return np.asarray(pcd.points)

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R

def apply_manual_pre_rotation(pcd):
    # 180° around Z, then 90° around X (adjust if needed)
    r = R.from_euler('zx', [180, 90], degrees=True)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    pcd.transform(T)
    return pcd


def crop_back_left_leg(pcd, x_thresh=7.3, y_thresh=-0.9):
    """
    Crop the back-left leg region from a point cloud based on observed GMM coords.

    Args:
        pcd (open3d.geometry.PointCloud): Input CAD point cloud.
        x_thresh (float): Points with x <= this will be removed.
        y_thresh (float): Points with y <= this will be removed.

    Returns:
        open3d.geometry.PointCloud: Cropped point cloud.
    """
    pts = np.asarray(pcd.points)
    
    # Remove points in the back-left corner
    mask = ~((pts[:, 0] <= x_thresh) & (pts[:, 1] <= y_thresh))

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])

    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return cropped

from scipy.spatial.transform import Rotation as R
import numpy as np

def rotate_about_centroid(pcd, euler_angles_deg, order='zx'):
    """Rotates the point cloud around its own centroid."""
    R_obj = R.from_euler(order, euler_angles_deg, degrees=True).as_matrix()
    center = pcd.get_center()
    
    T = np.eye(4)
    T[:3, :3] = R_obj

    # Translate to origin, rotate, then translate back
    pcd.translate(-center)
    pcd.transform(T)
    pcd.translate(center)

    return pcd

def plot_cad_alignment_stages(cad_before, cad_after, gmm_pc):
    fig = go.Figure()

    # GMM (orange)
    gmm_pts = np.asarray(gmm_pc.points)
    fig.add_trace(go.Scatter3d(
        x=gmm_pts[:, 0], y=gmm_pts[:, 1], z=gmm_pts[:, 2],
        mode='markers', marker=dict(size=3, color='orange'), name='GMM'
    ))

    # CAD before alignment (red)
    pts_before = np.asarray(cad_before.points)
    fig.add_trace(go.Scatter3d(
        x=pts_before[:, 0], y=pts_before[:, 1], z=pts_before[:, 2],
        mode='markers', marker=dict(size=2, color='red'), name='CAD (before ICP)'
    ))

    # CAD after alignment (green)
    pts_after = np.asarray(cad_after.points)
    fig.add_trace(go.Scatter3d(
        x=pts_after[:, 0], y=pts_after[:, 1], z=pts_after[:, 2],
        mode='markers', marker=dict(size=2, color='green'), name='CAD (after ICP)'
    ))

    fig.update_layout(
        title='CAD Alignment to GMM (Before and After)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        showlegend=True
    )
    
    fig.show()


def plot_cad_and_gmm(cad_pc, gmm_pc, cad_color='green', gmm_color='orange'):
    cad_pts = to_xyz_array(cad_pc)
    gmm_pts = to_xyz_array(gmm_pc)

    fig = go.Figure()

    # GMM means (yellow/orange)
    fig.add_trace(go.Scatter3d(
        x=gmm_pts[:, 0], y=gmm_pts[:, 1], z=gmm_pts[:, 2],
        mode='markers',
        marker=dict(size=3, color=gmm_color),
        name='GMM Means'
    ))

    # Aligned CAD points (green)
    fig.add_trace(go.Scatter3d(
        x=cad_pts[:, 0], y=cad_pts[:, 1], z=cad_pts[:, 2],
        mode='markers',
        marker=dict(size=2, color=cad_color),
        name='Aligned CAD'
    ))

    fig.update_layout(
        title="CAD Aligned to GMM",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True
    )

    fig.show()

import numpy as np

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

def pca_align(source_pts, target_pts):
    """
    Computes PCA alignment rotation matrix to align source to target.
    Both inputs should be (N, 3) numpy arrays.
    Returns a 3x3 rotation matrix.
    """
    # Center both point sets
    source_centered = source_pts - np.mean(source_pts, axis=0)
    target_centered = target_pts - np.mean(target_pts, axis=0)

    # Compute principal axes (PCA)
    U_s, _, _ = np.linalg.svd(source_centered.T @ source_centered)
    U_t, _, _ = np.linalg.svd(target_centered.T @ target_centered)

    # Compute rotation matrix from source → target
    R_align = U_t @ U_s.T

    # Ensure right-handedness
    if np.linalg.det(R_align) < 0:
        U_t[:, -1] *= -1
        R_align = U_t @ U_s.T

    return R_align

def center_and_scale_cad(cad_pc, gmm_pc):
    # Step 1: Estimate scale factor
    scale = estimate_scale(np.asarray(cad_pc.points), np.asarray(gmm_pc.points))
    cad_pc.scale(scale, center=cad_pc.get_center())

    # Step 2: Align centroids
    gmm_center = np.mean(np.asarray(gmm_pc.points), axis=0)
    cad_center = np.mean(np.asarray(cad_pc.points), axis=0)
    translation = gmm_center - cad_center
    cad_pc.translate(translation)

    return cad_pc

def print_T_calib(T_calib, limit=None):
    """
    Pretty print the T_calib matrices with rotation matrix, translation vector, and Euler angles.
    
    Args:
        T_calib (list): list of 4x4 np.ndarrays
        limit (int): number of matrices to print (optional)
    """
    n = len(T_calib) if limit is None else min(len(T_calib), limit)
    for i in range(n):
        T = T_calib[i]
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]
        euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)

        print(f"--- T_calib[{i}] ---")
        print("Rotation matrix:")
        print(R_mat)
        print("Translation vector:")
        print(t_vec)
        print("Euler angles (xyz, degrees):")
        print(euler)
        print()

def get_initial_positions(V, nObs, num_sensors):

    initial_positions = {}
    initial_positions["sensors"] = list(range(1, num_sensors + 1))
    initial_positions["num_obs"] = nObs // num_sensors
    initial_positions["pcd_list"] = []
    for idx in range(0, nObs):
        sensor_id = idx // initial_positions["num_obs"] 
        obs_id = idx % initial_positions["num_obs"] 
        initial_positions["pcd_list"].append({
            "file": f'{sensor_id}_{obs_id}',
            "points": V[idx].tolist()
        })
    return initial_positions

import plotly.graph_objects as go
import numpy as np

import os

import os
import numpy as np
import plotly.graph_objects as go

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_dashboard_all_views_2x4(initial_data, final_data, output_path="output/dashboard_2x4_views.png", width=2400, height=1200, scale=2):
    """
    Save a 2x4 Plotly dashboard showing all 4 views (top, side, front, iso) for initial and final observations.
    """

    def build_trace(pcd_list, num_obs):
        traces = []
        colors = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "purple", 5: "cyan"}

        for idx, pcd in enumerate(pcd_list):
            sensor_id = idx // num_obs
            pts = np.array(pcd["points"])
            color = colors.get(sensor_id, "gray")
            trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f"Sensor {sensor_id+1}",
                showlegend=False
            )
            traces.append(trace)
        return traces

    init_traces = build_trace(initial_data["pcd_list"], initial_data["num_obs"])
    final_traces = build_trace(final_data["pcd_list"], final_data["num_obs"])

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{'type': 'scene'}]*4, [{'type': 'scene'}]*4],
        subplot_titles=[
            "Initial (Top)", "Initial (Side)", "Initial (Front)", "Initial (Iso)",
            "Final (Top)", "Final (Side)", "Final (Front)", "Final (Iso)"
        ]
    )

    # View configs
    cameras = {
        "top": dict(eye=dict(x=0.001, y=0.001, z=2.5)),
        "side": dict(eye=dict(x=2.5, y=0.001, z=0.001)),
        "front": dict(eye=dict(x=0.001, y=2.5, z=0.001)),
        "iso": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    }

    views = ["top", "side", "front", "iso"]

    for i, view in enumerate(views):
        for trace in init_traces:
            fig.add_trace(trace, row=1, col=i+1)
        for trace in final_traces:
            fig.add_trace(trace, row=2, col=i+1)
        fig.update_scenes(camera=cameras[view], row=1, col=i+1)
        fig.update_scenes(camera=cameras[view], row=2, col=i+1)

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, scale=scale)
    print(f"Saved 2x4 dashboard to: {output_path}")

def crop_back_left_leg(pcd, x_thresh=8.0, y_thresh=-1.0):
    """
    Crop the back-left leg region from a point cloud based on observed GMM coords.

    Args:
        pcd (open3d.geometry.PointCloud): Input CAD point cloud.
        x_thresh (float): Points with x <= this will be removed.
        y_thresh (float): Points with y <= this will be removed.

    Returns:
        open3d.geometry.PointCloud: Cropped point cloud.
    """
    pts = np.asarray(pcd.points)
    
    # Remove points in the back-left corner
    mask = ~((pts[:, 0] <= x_thresh) & (pts[:, 1] <= y_thresh))

    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])

    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return cropped

def plot_all_observations_multiview(initial_positions, output_dir="output", basename="initial_positions"):
    os.makedirs(output_dir, exist_ok=True)
    sensors = initial_positions["sensors"]
    num_obs = initial_positions["num_obs"]
    pcd_list = initial_positions["pcd_list"]

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
            title=f"Initial Observations - {view_name.capitalize()} View",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
                camera=camera
            ),
            showlegend=False
        )

        output_path = os.path.join(output_dir, f"{basename}_{view_name}.png")
        fig.write_image(output_path, width=1200, height=800)
        print(f"Saved: {output_path}")


def estimate_scale(A, B):
    # A: sampled points from CAD
    # B: GMM means or LiDAR scan
    scale_A = np.linalg.norm(A.max(axis=0) - A.min(axis=0))
    scale_B = np.linalg.norm(B.max(axis=0) - B.min(axis=0))
    return scale_B / scale_A

def extract_back_left_leg(pcd, x_range=(7.8, 7.), y_range=(-1.4, -1.1), z_range=(0.0, 1.2)):
    """
    Extract the back-left leg region from a point cloud.

    Args:
        pcd (open3d.geometry.PointCloud): Input CAD point cloud.
        x_range (tuple): Range of X coordinates to keep.
        y_range (tuple): Range of Y coordinates to keep.
        z_range (tuple): Range of Z coordinates to keep.

    Returns:
        open3d.geometry.PointCloud: Filtered point cloud.
    """
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])
    
    if pcd.has_colors():
        filtered.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        filtered.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return filtered

def calibrate(data_path, config_file_path, sequence, num_iter=100, fixCentroids=False, num_points_param=400):
    pcds, num_sensors = generatePCDs.generate_data(data_path, config_file_path, sequence)
    Xin = create_gt.create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=num_points_param) + np.array([7.4, -1.0, 0.5])

    # Use deformable CAD mesh as initial shape
    shape_model = DeformableShapeModel("chair_simplified.obj")

    # Optional: center mesh or transform into approximate global frame
    center = shape_model.mesh.get_center()
    shape_model.mesh.translate(-center)
    shape_model.mesh.translate([7.5, -1.0, 0.5])
    shape_model.vertices = np.asarray(shape_model.mesh.vertices)  # ensure updated
    print(shape_model.vertices.shape)
    num_points = num_points_param  # or any N you want for your latent shape

    cad_pc = shape_model.mesh.sample_points_uniformly(number_of_points=num_points)
    cad_pts = np.asarray(cad_pc.points).astype(np.float64)

    V = [np.array(cloud.points) for cloud in pcds]

    nObs = len(V)
    print('nObs before batch', nObs)


    scale = estimate_scale(shape_model.vertices, V[0])
    print('scale: ' , scale)
    shape_model.mesh.scale(scale, center=shape_model.mesh.get_center())


    # Extract the rotated points
    cad_pc = shape_model.mesh.sample_points_uniformly(number_of_points=num_points)
    cad_pts = np.asarray(cad_pc.points).astype(np.float64)
    center = cad_pc.get_center()

    # Define the rotation: 90 degrees about X-axis
    r = R.from_euler('xz', [90,-45], degrees=True)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()

    # Apply the transformation: translate to origin, rotate, then translate back
    cad_pc.translate(-center)
    cad_pc.transform(T)
    cad_pc.translate(center)
    # cropped_cad = extract_back_left_leg(cad_pc)
    cad_pts = np.asarray(cad_pc.points).astype(np.float64)
    
    Xin = cad_pts.copy()
    
    batch_size = 0
    if batch_size:
        
        nObsPerSensor = nObs // num_sensors
        num_batches = math.ceil(nObsPerSensor / batch_size)
        print(num_batches)

        cur_batch = []
        merged_V = []
        for i in range (0, nObsPerSensor, batch_size):
            cur_batch_size = batch_size
            if i + batch_size >= nObsPerSensor:
                cur_batch_size = nObsPerSensor - i + 1
            merged_V.append(np.vstack(V[i:i+cur_batch_size]))
        for i in range (nObsPerSensor, nObsPerSensor * 2, batch_size):
            cur_batch_size = batch_size
            if i + batch_size >= nObsPerSensor * 2:
                cur_batch_size = nObsPerSensor * 2 - i + 1
            merged_V.append(np.vstack(V[i:i+cur_batch_size]))
        V = merged_V

    nObs = len(V)
    print(V[0].shape)


    # create socket client to publish display data
    display = True
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)
        display = params['display']
    
    if display:
        client = SocketIOClient('http://127.0.0.1:5000')
    else:
        client = None
    # client = None

    # send initial point cloud positions (vehicle frame)
    initial_positions = get_initial_positions(V, nObs, num_sensors)
    
    save_images = False
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)
        save_images = params['save_images']
    
    if save_images:
        plot_all_observations_multiview(initial_positions)
    
    if client:
        client.emit("initial_positions", initial_positions)

        # send initial gmm means
        client.emit("gmm_means", {"Xin": Xin.tolist(), "X": Xin.tolist(), "num_iter": 0})

    print("####### Perform Calibration and Model Generation. ########")
    X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=num_iter, socket_client=client, num_sensors=num_sensors, fixCentroids=fixCentroids, save_images=save_images)
 
    final_registrations = get_registrations(TV, len(TV), num_sensors)
    
    T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
    print_T_calib(T_calib)
    T_final = transformPCDs.mean_transform(T_calib)
    # Extract rotation matrix (upper-left 3x3 part)
    R_calib_new = T_final[:3, :3]

    # Compute Euler angles (roll, pitch, yaw) using intrinsic XYZ convention
    rotation_new = R.from_matrix(R_calib_new)
    euler_angles_rad_new = rotation_new.as_euler('xyz', degrees=False)  # Radians
    euler_angles_deg_new = rotation_new.as_euler('xyz', degrees=True)
    print('euler angles (deg)', euler_angles_deg_new)
    print('euler angles (rad)', euler_angles_rad_new)

    translation_new = T_final[:3, 3]
    print('Translation (x, y, z):', translation_new)
    
    euler_angles_deg_before = euler_angles_deg_new
    translation_before = translation_new 

    # validation

    print("Calibration Error: \n")
    print(T_final)
    gmmcalib_result = [T_final, X]
    with open("/app/output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump(gmmcalib_result, f) 

    # Step 1: Convert GMM (X.T) to point cloud
    gmm_pc = o3d.geometry.PointCloud()
    gmm_pc.points = o3d.utility.Vector3dVector(X.T)

    # Step 2: Align CAD mesh to GMM via ICP
    cad_pc = shape_model.mesh.sample_points_uniformly(500)
    icp_result = o3d.pipelines.registration.registration_icp(
        source=cad_pc,
        target=gmm_pc,
        max_correspondence_distance=0.1,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    shape_model.mesh.transform(icp_result.transformation)

    # Before deformation
    cad_pts = np.asarray(shape_model.mesh.sample_points_uniformly(number_of_points=500).points)
    gmm_means = X
    gmm_pts = gmm_means.T

    # PCA or ICP alignment
    Rt = pca_align(cad_pts, gmm_pts)
    T = np.eye(4)
    T[:3, :3] = Rt
    shape_model.mesh.transform(T)

    deformed_mesh = cad_deform_to_gmm(shape_model, X)
    deformed_pc = deformed_mesh.sample_points_uniformly(number_of_points=500)
    # points = np.asarray(shape_model.mesh.sample_points_uniformly(number_of_points=500).points) 
    # X = points.T
    # X = np.asarray(deformed_pc.points).T
    pts = np.asarray(deformed_mesh.sample_points_uniformly(500).points)
    print("Sampled points stats:\n", np.min(pts, axis=0), np.max(pts, axis=0))

    print('X', X.shape)

    # refine = False
    # if refine:
    #     print("\n### Aligning CAD model to GMM shape ###")

    #     # Step 1: Convert GMM means (X) to Open3D point cloud
    #     X = X.T
    #     gmm_pc = o3d.geometry.PointCloud()
    #     gmm_pc.points = o3d.utility.Vector3dVector(X)
    #     print('X shape', X.shape)

    #     import copy

    #     # Get numpy arrays from point clouds
    #     cad_pts = np.asarray(cad_pc.points)
    #     gmm_pts = np.asarray(gmm_pc.points)

    #     # Get PCA rotation matrix
    #     R_pca = pca_align(cad_pts, gmm_pts)

    #     # Build transform
    #     T = np.eye(4)
    #     T[:3, :3] = R_pca
    #     center = cad_pc.get_center()

    #     # Rotate CAD about its own center using PCA
    #     cad_pc.translate(-center)
    #     cad_pc.transform(T)
    #     cad_pc.translate(center)

    #     cad_before = copy.deepcopy(cad_pc)
    #     cad_pc = center_and_scale_cad(cad_pc, gmm_pc)


    #     # Step 3: Run ICP
    #     threshold = 0.1
    #     icp_result = o3d.pipelines.registration.registration_icp(
    #         source=cad_pc,
    #         target=gmm_pc,
    #         max_correspondence_distance=threshold,
    #         init=np.eye(4),
    #         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #     )

    #     T_cad_to_gmm = icp_result.transformation
    #     print("ICP Fitness:", icp_result.fitness)
    #     print("ICP RMSE:", icp_result.inlier_rmse)
    #     print("Transformation (CAD → GMM):\n", T_cad_to_gmm)

    #     # Step 4: Apply transform to get aligned CAD
    #     import copy

    #     cad_after = copy.deepcopy(cad_pc)
    #     cad_after.transform(T_cad_to_gmm)

    #     # Call it
    #     plot_cad_alignment_stages(cad_before, cad_after, gmm_pc)


    #     Xin = np.asarray(cad_after.points).astype(np.float64)
        
    #     initial_positions = get_initial_positions(V, nObs, num_sensors)
    #     if client:
    #         client.emit("initial_positions", initial_positions)

    #         # send initial gmm means
    #         client.emit("gmm_means", {"Xin": Xin.tolist(), "X": Xin.tolist(), "num_iter": 0})

    #     X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=num_iter, socket_client=client, num_sensors=num_sensors, fixCentroids=True)
    
    #     T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    #     T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    #     T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
    #     print_T_calib(T_calib)
    #     T_final = transformPCDs.mean_transform(T_calib)
    #     # Extract rotation matrix (upper-left 3x3 part)
    #     R_calib_new = T_final[:3, :3]

    #     # Compute Euler angles (roll, pitch, yaw) using intrinsic XYZ convention
    #     rotation_new = R.from_matrix(R_calib_new)
    #     euler_angles_rad_new = rotation_new.as_euler('xyz', degrees=False)  # Radians
    #     euler_angles_deg_new = rotation_new.as_euler('xyz', degrees=True)
    #     print('euler angles (deg) before', euler_angles_deg_before)
    #     print('euler angles (rad)', euler_angles_rad_new)
    #     print('euler angles (deg)', euler_angles_deg_new)
    #     print('Translation (x, y, z) before:', translation_before)
    #     print('euler angles (rad)', euler_angles_rad_new)
    
    return T_final, initial_positions, final_registrations, Xin, X.T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration script")
    parser.add_argument("--data_path", type=str, help="Path to data", default="../data")
    parser.add_argument("--config_file_path", type=str, help="Path to config file", default="../config/config.yaml")
    parser.add_argument("--sequence", nargs='+', type=int, help="Sequence sequence of pcds")

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file_path))

    if args.sequence is None:
        sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    else:
        sequence = args.sequence

    calibrate(data_path, config_file_path, sequence)
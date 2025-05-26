import numpy as np
import open3d as o3d
from collections import defaultdict

def compute_vertex_point_weights(mesh_model, pcd, radius=0.1, k=3):
    """
    For each point in the point cloud, find its nearest mesh vertices.
    Return: dict {vertex_idx: list of (point, weight)}
    """
    vertex_to_points = defaultdict(list)

    # Use Open3D KDTree for fast nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(mesh_model.to_point_cloud())

    points = np.asarray(pcd.points)

    for point in points:
        [_, idxs, dists] = kdtree.search_radius_vector_3d(point, radius)
        if len(idxs) == 0:
            continue

        # Optional: use k nearest only (up to k), and compute weights
        for idx, dist in zip(idxs[:k], dists[:k]):
            weight = np.exp(-dist / (radius ** 2))  # simple Gaussian weight
            if weight < 0.01:
                continue

            vertex_to_points[idx].append((point, weight))

    return vertex_to_points

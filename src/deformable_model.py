import open3d as o3d
import numpy as np
from correspondence import compute_vertex_point_weights
from deformation import compute_vertex_targets, apply_deformation_with_shape_prior
from procrustes import compute_best_fit_transform

class DeformableShapeModel:
    def __init__(self, obj_path, scale=0.01):
        self.mesh = o3d.io.read_triangle_mesh(obj_path)
        self.mesh.compute_vertex_normals()
        self.mesh.scale(scale, center=self.mesh.get_center())
        self.vertices = np.asarray(self.mesh.vertices).copy()
        self.original_vertices = self.vertices.copy()

    def update_vertices(self, new_vertices):
        self.vertices = new_vertices
        self.mesh.vertices = o3d.utility.Vector3dVector(new_vertices)

    def deform_to_pointcloud(self, pcd, radius=0.1, alpha=0.2, lambda_shape=0.1):
        correspondences = compute_vertex_point_weights(self, pcd, radius=radius)
        targets = compute_vertex_targets(
            correspondences,
            num_vertices=len(self.vertices),
            original_vertices=self.vertices
        )
        apply_deformation_with_shape_prior(self, targets, alpha=alpha, lambda_shape=lambda_shape)
        return correspondences

    def update_sensor_transform(self, pcd, correspondences):
        points, targets, weights = [], [], []
        for vertex_idx, matches in correspondences.items():
            v = self.vertices[vertex_idx]
            for p, w in matches:
                points.append(p)
                targets.append(v)
                weights.append(w)
        points, targets, weights = map(np.array, (points, targets, weights))
        T = compute_best_fit_transform(points, targets, weights)
        pcd.transform(T)
        return T

    def to_geometry(self):
        return self.mesh

import numpy as np
import open3d as o3d

def compute_vertex_targets(vertex_to_points, num_vertices, original_vertices):
    """
    Returns an (N, 3) array of target positions.
    If no data for a vertex, fall back to its original position.
    """
    target_positions = np.zeros_like(original_vertices)

    for idx in range(num_vertices):
        correspondences = vertex_to_points.get(idx, [])
        if not correspondences:
            target_positions[idx] = original_vertices[idx]
        else:
            pts, weights = zip(*correspondences)
            pts = np.array(pts)
            weights = np.array(weights)
            weights = weights / weights.sum()  # normalize
            target_positions[idx] = np.sum(weights[:, None] * pts, axis=0)

    return target_positions

def apply_deformation(mesh_model, target_positions, alpha=0.2):
    """
    Move each vertex toward its target position by a small step (alpha).
    Updates mesh_model.vertices in place.
    """
    current = mesh_model.vertices
    updated = (1 - alpha) * current + alpha * target_positions
    mesh_model.vertices = updated
    mesh_model.mesh.vertices = o3d.utility.Vector3dVector(updated)
    
def apply_deformation_with_shape_prior(mesh_model, target_positions, alpha=0.2, lambda_shape=0.1):
    V = mesh_model.vertices
    V0 = mesh_model.original_vertices
    T = target_positions

    # Move toward data (T) and original shape (V0)
    updated = (1 - alpha) * V + alpha * (T + lambda_shape * V0) / (1 + lambda_shape)
    mesh_model.vertices = updated
    mesh_model.mesh.vertices = o3d.utility.Vector3dVector(updated)
    
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import open3d as o3d

def cad_deform_to_gmm(shape_model, gmm_means, weight_data=1.0, weight_smooth=1e2):
    mesh = shape_model.mesh
    verts = np.asarray(mesh.vertices)       # (V, 3) actual mesh vertices
    faces = np.asarray(mesh.triangles)

    # Build correspondence from mesh vertices → GMM means
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    _, indices = tree.query(gmm_means.T)    # gmm_means shape: (3, N), so transpose to (N, 3)

    target_pts = gmm_means.T  # Final shape: (N, 3), corresponds to `indices`

    # Step 2: Build Laplacian (uniform umbrella for simplicity)
    n_verts = verts.shape[0]
    A = scipy.sparse.lil_matrix((n_verts, n_verts))

    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i+1)%3]
            A[v1, v2] = 1
            A[v2, v1] = 1

    D = scipy.sparse.diags(A.sum(axis=1).A1)
    L = D - A  # Graph Laplacian

    # Step 3: Build energy matrix
    n_constraints = len(indices)
    I_data = scipy.sparse.identity(n_verts)

    A_data = scipy.sparse.lil_matrix((n_constraints, n_verts))
    for i, idx in enumerate(indices):
        A_data[i, idx] = 1

    A_total = scipy.sparse.vstack([
        weight_data * A_data,
        weight_smooth * L
    ])
    
    B_total = np.vstack([
        weight_data * target_pts,
        np.zeros((n_verts, 3))
    ])
    
    print("Deforming", len(indices), "vertices to match GMM points")
    print("Avg dist to target:", np.linalg.norm(verts[indices] - target_pts, axis=1).mean())


    # Step 4: Solve least squares
    V_deformed = np.zeros_like(verts)
    for dim in range(3):
        x = scipy.sparse.linalg.lsqr(A_total, B_total[:, dim])[0]
        V_deformed[:, dim] = x

    # Step 5: Assign to mesh and return
    import copy
    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.vertices = o3d.utility.Vector3dVector(V_deformed)
    return deformed_mesh


import numpy as np
import open3d as o3d
import pandas as pd

import numpy as np
import pandas as pd

def load_cube_model(file_path):
    """Load the simulated cube model as a point cloud."""
    cube_df = pd.read_csv(file_path)
    return cube_df.to_numpy().T  # Return as (3, N) array for consistency


def create_cube(side_length=1.0, center=(0, 0, 0), density=50):
    """Simulate a cube as a point cloud."""
    half_side = side_length / 2
    lin_space = np.linspace(-half_side, half_side, density)

    # Generate the 6 faces of the cube
    faces = []

    # XY planes
    for z in [-half_side, half_side]:
        X, Y = np.meshgrid(lin_space, lin_space)
        Z = np.full_like(X, z)
        faces.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # XZ planes
    for y in [-half_side, half_side]:
        X, Z = np.meshgrid(lin_space, lin_space)
        Y = np.full_like(X, y)
        faces.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # YZ planes
    for x in [-half_side, half_side]:
        Y, Z = np.meshgrid(lin_space, lin_space)
        X = np.full_like(Y, x)
        faces.append(np.stack([X, Y, Z], axis=-1).reshape(-1, 3))

    # Combine all faces
    cube_points = np.vstack(faces)

    # Apply translation to center the cube
    cube_points += np.array(center)

    return cube_points

# Create a cube point cloud
cube_points = create_cube(side_length=1.0, center=(0, 0, 0), density=50)

cube_df = pd.DataFrame(cube_points, columns=["X", "Y", "Z"])
# Save the cube point cloud to a CSV file
cube_file_path = "./simulated_cube_point_cloud.csv"
cube_df.to_csv(cube_file_path, index=False)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot the simulated cube using Matplotlib
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Save the cube visualization as an image using Matplotlib (without displaying it)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cube points
ax.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2], s=1, alpha=0.6)

# Set plot limits for better visualization
ax.set_xlim([-0.6, 0.6])
ax.set_ylim([-0.6, 0.6])
ax.set_zlim([-0.6, 0.6])

# Set axis labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Simulated Cube Point Cloud')

# Save the plot as an image file
plot_file_path = "./simulated_cube_point_cloud.png"
plt.savefig(plot_file_path)
plt.close()

plot_file_path

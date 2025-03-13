import open3d as o3d

# Load the PCD file
pcd_file = "./point_clouds_overlap/131_frontleftwithnoise001.pcd"  # Replace with your actual file path
pcd = o3d.io.read_point_cloud(pcd_file)

# Print basic information
print(pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="PCD Viewer")

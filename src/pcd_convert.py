import numpy as np
import open3d as o3d

def txt_to_pcd(input_txt_file, output_pcd_file):
    """Convert a space-separated text file with x, y, z points to PCD format."""
    
    # Load text file (assuming space-separated values)
    points = np.loadtxt(input_txt_file)

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save as PCD
    o3d.io.write_point_cloud(output_pcd_file, pcd)
    print(f"Saved {output_pcd_file}")

# Example usage:
for i in range(1,5):
    txt_to_pcd(f'bunny_data/cutView{i}.txt', f'bunny_data/sensor_{i}/1.pcd')

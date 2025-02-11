import open3d as o3d
import numpy as np
import copy

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_raw = o3d.io.read_point_cloud("./cube_data/sensor_1/1.pcd")
    target_raw = o3d.io.read_point_cloud("./cube_data/sensor_2/1.pcd")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_raw)
    vis.add_geometry(target_raw)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    while True:
        vis.update_geometry(source_raw)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
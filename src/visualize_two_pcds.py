#!/usr/bin/env python3
"""
visualize_two_pcds.py

Load two PCD files (e.g., sensor0_raw.pcd and sensor1_aligned.pcd) and display them together
in Open3D with different colors so you can inspect their relative alignment.

Usage:
    python visualize_two_pcds.py \
        --pcd1 aligned_pcds/sensor0_raw.pcd \
        --pcd2 aligned_pcds/sensor1_aligned.pcd
"""

import argparse
import open3d as o3d
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Load and display two PCDs together with distinct colors."
    )
    parser.add_argument(
        "--pcd1",
        type=str,
        required=True,
        help="Path to the first PCD file (e.g. sensor0_raw.pcd)."
    )
    parser.add_argument(
        "--pcd2",
        type=str,
        required=True,
        help="Path to the second PCD file (e.g. sensor1_aligned.pcd)."
    )
    parser.add_argument(
        "--color1",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        help="RGB color for the first cloud (three floats in [0,1], default is red)."
    )
    parser.add_argument(
        "--color2",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="RGB color for the second cloud (three floats in [0,1], default is blue)."
    )
    parser.add_argument(
        "--point_size",
        type=int,
        default=2,
        help="Size of points in the visualizer (default 2)."
    )
    args = parser.parse_args()

    # Load the two PCDs
    pcd1 = o3d.io.read_point_cloud(args.pcd1)
    if pcd1.is_empty():
        raise RuntimeError(f"Failed to load or empty point cloud: {args.pcd1}")

    pcd2 = o3d.io.read_point_cloud(args.pcd2)
    if pcd2.is_empty():
        raise RuntimeError(f"Failed to load or empty point cloud: {args.pcd2}")

    # Assign colors
    color1 = np.array(args.color1, dtype=float).reshape(1, 3)
    color2 = np.array(args.color2, dtype=float).reshape(1, 3)

    pcd1.paint_uniform_color(color1.flatten())
    pcd2.paint_uniform_color(color2.flatten())

    # Create a single coordinate frame at origin (optional)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # Configure visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Two PCD Alignment", width=1024, height=768)

    render_opt = vis.get_render_option()
    # render_opt.background_color = np.array([0.05, 0.05, 0.05])
    # render_opt.point_size = args.point_size

    # Add geometries
    vis.add_geometry(axis)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    # Optionally, you can set up the view control here (camera angles).
    ctr = vis.get_view_control()
    # For instance, you can get default parameters and slightly adjust them:
    # param = ctr.convert_to_pinhole_camera_parameters()
    # param.extrinsic = np.array([
    #     [1.0, 0.0,  0.0, 0.0],
    #     [0.0, 1.0,  0.0, -2.0],  # shift camera back along Y
    #     [0.0, 0.0,  1.0,  2.0],  # raise camera above Z
    #     [0.0, 0.0,  0.0,  1.0]
    # ])
    # ctr.convert_from_pinhole_camera_parameters(param)

    print("Press ‘Q’ or ‘Esc’ in the window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

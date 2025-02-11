import open3d as o3d
import numpy as np
import copy

class Live3DVisualizer:
    def __init__(self, static_geometry):
        """
        Initializes a non-blocking Open3D visualizer with:
        - One static geometry (e.g., a reference object).
        - One dynamic geometry (e.g., a moving point cloud).
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # Store and add static geometry (unchanging)
        self.static_geometry = static_geometry
        self.vis.add_geometry(self.static_geometry)

        # Create and add dynamic geometry (this will be updated later)
        self.dynamic_geometry = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.dynamic_geometry)
        self.vis.register_key_callback(ord("Q"), self.quit_visualization)

        # Colorize PCDs for distinction
        self.static_geometry.paint_uniform_color([1, 0, 0])  # Red for Folder 1
        self.dynamic_geometry.paint_uniform_color([0, 0, 1])  # Blue for Folder 2

        self.vis.poll_events()
        self.vis.update_renderer()

    def update_dynamic_geometry(self, new_geometry):
        """
        Updates the dynamic geometry in the visualizer.
        
        Parameters:
        - new_geometry (o3d.geometry.PointCloud): The new point cloud to display.
        """
        self.dynamic_geometry.points = new_geometry.points
        if new_geometry.has_colors():
            self.dynamic_geometry.colors = new_geometry.colors
        if new_geometry.has_normals():
            self.dynamic_geometry.normals = new_geometry.normals

        self.dynamic_geometry.paint_uniform_color([0, 0, 1])  # Blue for Folder 2

        self.vis.update_geometry(self.dynamic_geometry)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Closes the Open3D visualizer."""
        self.vis.destroy_window()

    def quit_visualization(self, vis):
        """Exit the Open3D visualization."""
        print("Exiting visualization...")
        self.vis.destroy_window()
        exit(0)
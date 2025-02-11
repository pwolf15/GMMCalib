import open3d as o3d
import os
import glob

class PCDVisualizer:
    def __init__(self, pcd_folder):
        """Initialize the Open3D visualizer to step through PCD files."""
        self.pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
        if not self.pcd_files:
            print("No .pcd files found in the specified directory.")
            exit(1)

        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        
        # Load first PCD
        self.pcd = o3d.io.read_point_cloud(self.pcd_files[self.index])
        self.vis.add_geometry(self.pcd)

        # Register key callbacks
        self.vis.register_key_callback(ord("N"), self.next_pcd)
        self.vis.register_key_callback(ord("B"), self.prev_pcd)
        self.vis.register_key_callback(ord("Q"), self.quit_visualization)

        print(f"Press 'N' for next, 'B' for previous, 'Q' to quit.")

    def next_pcd(self, vis):
        """Load the next PCD file."""
        if self.index < len(self.pcd_files) - 1:
            self.index += 1
            self.update_pcd()
    
    def prev_pcd(self, vis):
        """Load the previous PCD file."""
        if self.index > 0:
            self.index -= 1
            self.update_pcd()

    def quit_visualization(self, vis):
        """Exit the Open3D visualization."""
        print("Exiting visualization...")
        self.vis.destroy_window()
        exit(0)

    def update_pcd(self):
        """Update the PCD displayed in Open3D."""
        self.pcd.points = o3d.io.read_point_cloud(self.pcd_files[self.index]).points
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        print(f"Displaying: {self.pcd_files[self.index]} ({self.index + 1}/{len(self.pcd_files)})")

    def run(self):
        """Run the Open3D visualizer loop."""
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()

# Run visualization
if __name__ == "__main__":
    folder = input("Enter the folder path containing PCD files: ").strip()
    visualizer = PCDVisualizer(folder)
    visualizer.run()

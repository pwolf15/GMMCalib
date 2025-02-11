import open3d as o3d
import os
import glob

class DualPCDVisualizer:
    def __init__(self, folder1, folder2):
        """
        Initialize the Open3D visualizer to step through synchronized PCD files from two folders.
        
        Parameters:
        - folder1 (str): Path to the first folder containing .pcd files.
        - folder2 (str): Path to the second folder containing .pcd files.
        """
        self.pcd_files1 = sorted(glob.glob(os.path.join(folder1, "*.pcd")))
        self.pcd_files2 = sorted(glob.glob(os.path.join(folder2, "*.pcd")))

        if len(self.pcd_files1) != len(self.pcd_files2):
            print("⚠️ Warning: The number of PCD files in the two folders is not the same!")

        self.total_files = min(len(self.pcd_files1), len(self.pcd_files2))

        if self.total_files == 0:
            print("No .pcd files found in the specified directories.")
            exit(1)

        self.index = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # Load the first pair of PCDs
        self.pcd1 = o3d.io.read_point_cloud(self.pcd_files1[self.index])
        self.pcd2 = o3d.io.read_point_cloud(self.pcd_files2[self.index])

        # Colorize PCDs for distinction
        self.pcd1.paint_uniform_color([1, 0, 0])  # Red for Folder 1
        self.pcd2.paint_uniform_color([0, 0, 1])  # Blue for Folder 2

        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)

        # Register key callbacks * 1.1
        self.vis.register_key_callback(ord("N"), self.next_pcd)
        self.vis.register_key_callback(ord("B"), self.prev_pcd)
        self.vis.register_key_callback(ord("Q"), self.quit_visualization)
        self.vis.register_key_callback(ord("Z"), self.zoom_in)  # Press "Z" to zoom in
        self.vis.register_key_callback(ord("X"), self.zoom_out)  # Press "X" to zoom out
        self.zoom_factor = 1.0
        print(f"Press 'N' for next, 'B' for previous, 'Q' to quit.")

    def next_pcd(self, vis):
        """Load the next pair of PCD files."""
        if self.index < self.total_files - 1:
            self.index += 1
            self.update_pcd()

    def prev_pcd(self, vis):
        """Load the previous pair of PCD files."""
        if self.index > 0:
            self.index -= 1
            self.update_pcd()

    def quit_visualization(self, vis):
        """Exit the Open3D visualization."""
        print("Exiting visualization...")
        self.vis.destroy_window()
        exit(0)

    def update_pcd(self):
        """Update the displayed PCDs in Open3D."""
        print(f"Displaying: {self.pcd_files1[self.index]} & {self.pcd_files2[self.index]} ({self.index + 1}/{self.total_files})")

        # Load new PCDs
        self.pcd1.points = o3d.io.read_point_cloud(self.pcd_files1[self.index]).points
        self.pcd2.points = o3d.io.read_point_cloud(self.pcd_files2[self.index]).points

        # Keep colors for distinction
        self.pcd1.paint_uniform_color([1, 0, 0])  # Red for Folder 1
        self.pcd2.paint_uniform_color([0, 0, 1])  # Blue for Folder 2

        # Update visualization
        self.vis.update_geometry(self.pcd1)
        self.vis.update_geometry(self.pcd2)
        self.vis.poll_events()
        self.vis.update_renderer()

    def zoom_in(self, vis):
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=-5.0) # Decrease zoom  
        return False  # Keeps visualization running

    def zoom_out(self, vis):
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=5.0) # Decrease zoom  
        return False  # Keeps visualization running
        
    def run(self):
        """Run the Open3D visualizer loop."""
        while True:
            self.vis.poll_events()
            self.vis.update_renderer()

# Run visualization
if __name__ == "__main__":
    base_folder = './data/'
    folder1 = f'{base_folder}/sensor_1' # input("Enter the first folder path containing PCD files: ").strip()
    folder2 = f'{base_folder}/sensor_2' #input("Enter the second folder path containing PCD files: ").strip()
    
    visualizer = DualPCDVisualizer(folder1, folder2)
    visualizer.run()

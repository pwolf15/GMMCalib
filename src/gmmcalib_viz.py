import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
# from modelgenerator_af import jgmm_af
import create_gt
import numpy as np
import pickle

import numpy as np
import pandas as pd

import open3d as o3d
# from visualizer import Live3DVisualizer
# from visualizer_plotly import Live3DVisualizerPlotly
from server import WebSocketServer
import yaml
import threading

import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
import create_gt
import numpy as np
import pickle
import open3d as o3d
from server import WebSocketServer
import yaml
import threading

class Calibration:
    def __init__(self, data_path, config_path, V, Xin):
        """Initialize Calibration process and WebSocket Server."""
        self.data_path = data_path
        self.config_path = config_path
        self.V = V
        self.n = len(self.V) // 2
        self.Xin = Xin  # ✅ Store initial point cloud
        self.X = np.copy(Xin)  # ✅ Initially, X is equal to Xin

        self.ws_server = WebSocketServer(calibration_instance=self)  # ✅ Start WebSocket Server
        self.ws_server.start_server()

    def get_initial_calib_data(self):
        """Return Xin and X as the initial dynamic point cloud data."""
        return {
            "Xin": {
                "x": self.Xin[:, 0].tolist(),
                "y": self.Xin[:, 1].tolist(),
                "z": self.Xin[:, 2].tolist(),
            },
            "X": {
                "x": self.X[:, 0].tolist(),
                "y": self.X[:, 1].tolist(),
                "z": self.X[:, 2].tolist(),
            }
        }

    def process_results(self, TV, AllT, V):
        """Process results after jgmm and send to WebSocket."""
        T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(len(V) // 2)]
        T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(len(V) // 2, len(V))]

        T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
        T_final = transformPCDs.mean_transform(T_calib)

        print("Calibration Error: \n", T_final)

        # ✅ Save results
        with open("/app/output/gmmcalib_result.pkl", "wb") as f:
            pickle.dump([T_final, V], f)

    def run_jgmm_async(self, V, Xin):
        """Runs `jgmm()` in a background thread and sends updates to WebSocket."""
        def run_jgmm():
            print("####### Running jgmm asynchronously... ########")
            X, TV, AllT, pk = jgmm(V=V, Xin=Xin, maxNumIter=50, server=self.ws_server)

            # ✅ Send final results to WebSocket
            self.process_results(TV, AllT, V)

        threading.Thread(target=run_jgmm, daemon=True).start()  # ✅ Run in background

    def read_config(self):
        """Read and return the configuration parameters."""
        data_path = self.data_path  # ✅ Replace with actual path
        config_file_path = self.config_path  # ✅ Replace with actual path
        transform_sensor_1, transform_sensor_2, min_bound, max_bound, number_of_sensors = read_config(data_path, config_file_path)

        return {
            "data_path": data_path,
            "config_file_path": config_file_path,
            "transform_sensor_1": transform_sensor_1,
            "transform_sensor_2": transform_sensor_2,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "number_of_sensors": number_of_sensors
        }

    def get_static_data(self):
        """Return static point cloud data as pairs."""
        point_cloud_pairs = []

        for i in range(self.n):
            point_cloud_pairs.append({
                "pair_id": i,
                "sensor_1": {
                    "x": self.V[i][:, 0].tolist(),
                    "y": self.V[i][:, 1].tolist(),
                    "z": self.V[i][:, 2].tolist()
                },
                "sensor_2": {
                    "x": self.V[self.n + i][:, 0].tolist(),
                    "y": self.V[self.n + i][:, 1].tolist(),
                    "z": self.V[self.n + i][:, 2].tolist()
                }
            })

        return point_cloud_pairs

def read_config(data_path, config_file_path):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_sensor_1 = config_data.get("transform_sensor_1", "")[0]
    transform_sensor_2 = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]
    number_of_sensors = config_data.get("number_of_sensors", "")

    return transform_sensor_1, transform_sensor_2, min_bound, max_bound, number_of_sensors

def get_initial_pcd_figure(pcds_vf):
    num_steps = len(pcds_vf) // 2 # number of time steps
    print(f'Num steps: {num_steps}')
    traces = []
    for i in range(num_steps):
        traces.append({"points": pcds_vf[i], "color": "red", "size": 3, "name": f"sensor 1 step {i}", "is_static": False })
        traces.append({"points": pcds_vf[num_steps + i], "color": "blue", "size": 3, "name": f"sensor 2 step {i}", "is_static": False})
    figures_config = {
        "title": "Initial position of the point clouds",
        "traces": traces
    }
    return figures_config

def get_figures_config():

    figures_config = [
        {
            "title": "Initial position of the point clouds",
            "traces": [
                {"points": np.random.rand(100, 3), "color": "red", "size": 3, "name": "Xin", "is_static": False},
                {"points": np.random.rand(50, 3), "color": "blue", "size": 3, "name": "X", "is_static": False}
            ]
        },
        {
            "title": "Centroids after X iterations",
            "traces": [
                {"points": np.random.rand(100, 3), "color": "red", "size": 3, "name": "Xin", "is_static": False},
                {"points": np.random.rand(50, 3), "color": "blue", "size": 3, "name": "X", "is_static": False}
            ]
        },
        # {
        #     "title": "Registration of the sets after X iterations",
        #     "traces": [
        #         {"points": np.random.rand(100, 3), "color": "green", "size": 3, "name": "Static Trace 2", "is_static": True},
        #         {"points": np.random.rand(50, 3), "color": "purple", "size": 3, "name": "Dynamic Trace 2", "is_static": False}
        #     ]
        # }
    ]
    return figures_config


def calibrate(data_path, config_file_path, sequence):
    pcds = generatePCDs.generate_data(data_path, config_file_path, sequence)
    Xin = create_gt.create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=400) + np.array([9.8, 4.75, 0.38])
    # cube_model_path = "./simulated_cube_point_cloud.csv"
    # Xin = load_cube_model(cube_model_path)

    # inject error

    V = [np.array(cloud.points) for cloud in pcds]
    print(V)
    nObs = len(V)
    print(nObs)

    print("####### Perform Calibration and Model Generation. ########")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Xin)

    # get initial figure config
    figures_config = get_figures_config()

    # initial plot
    figures_config[0] = get_initial_pcd_figure(V)
    print(figures_config[0])
    transform_sensor_1, transform_sensor_2, min_bound, max_bound, number_of_sensors = read_config(data_path, config_file_path)
    
    def run_jgmm(visualizer):
        """Function to execute jgmm asynchronously."""
        pcds = generatePCDs.generate_data(data_path, config_file_path, sequence)
        Xin = create_gt.create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=400) + np.array([9.8, 4.75, 0.38])
        V = [np.array(cloud.points) for cloud in pcds]
        nObs = len(V)

        print("####### Running jgmm in separate thread... ########")
        visualizer.update_dynamic_geometry(1, 0, Xin)
        visualizer.update_dynamic_geometry(1, 1, Xin)
        X, TV, AllT, pk = jgmm(V=V, Xin=Xin, maxNumIter=100, visualizer=visualizer)

        T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
        T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

        T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
        T_final = transformPCDs.mean_transform(T_calib)
        print("Calibration Error: \n")
        print(T_final)
        visualizer.calib_metadata["T_final"] = T_final

        # ✅ After `jgmm` completes, update the visualization
        print("####### jgmm completed! ########")

    # ✅ Start WebSocket Server inside Calibration
    calibration = Calibration(data_path, config_file_path, V, Xin)

    # ✅ Run `jgmm` asynchronously and send data to WebSocket
    calibration.run_jgmm_async(V, Xin)
    
    # visualizer = Live3DVisualizerPlotly(figures_config, {
    #     "config": config_file_path,
    #     "data": data_path,
    #     "min_bound": min_bound,
    #     "max_bound": max_bound,
    #     "transform_sensor_1": transform_sensor_1,
    #     "transform_sensor_2": transform_sensor_2,
    #     "T_final": {},
    #     "run_jgmm": run_jgmm,
    #     "has_started": False
    # })
    # visualizer.update_dynamic_geometry(1, 0, Xin)
    # visualizer.update_dynamic_geometry(1, 1, Xin)

    # X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=50)
    # print(len(TV))
    # print(len(AllT))
    while True:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration script")
    parser.add_argument("--data_path", type=str, help="Path to data", default="../data")
    parser.add_argument("--config_file_path", type=str, help="Path to config file", default="../config/config.yaml")
    parser.add_argument("--sequence", nargs='+', type=int, help="Sequence sequence of pcds")

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file_path))

    print(f'Config path: {config_file_path}')
    if args.sequence is None:
        sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    else:
        sequence = args.sequence

    calibrate(data_path, config_file_path, sequence)

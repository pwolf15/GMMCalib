import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
# from modelgenerator_af import jgmm_af
import create_gt
import numpy as np
import pickle
import generateMesh

import numpy as np
import pandas as pd

import open3d as o3d
from visualizer import Live3DVisualizer
from visualizer_plotly import Live3DVisualizerPlotly

def load_cube_model(file_path):
    """Load the simulated cube model as a point cloud."""
    cube_df = pd.read_csv(file_path)
    return cube_df.to_numpy().T  # Return as (3, N) array for consistency

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
                {"points": np.random.rand(100, 3), "color": "red", "size": 3, "name": "Xin", "is_static": True},
                {"points": np.random.rand(50, 3), "color": "blue", "size": 3, "name": "X", "is_static": False}
            ]
        },
        {
            "title": "Centroid after X iterations",
            "traces": [
                {"points": np.random.rand(100, 3), "color": "red", "size": 3, "name": "Xin", "is_static": True},
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
    visualizer = Live3DVisualizerPlotly(figures_config)
    visualizer.update_dynamic_geometry(1, 0, Xin)
    X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=50, visualizer=visualizer)
    print(len(TV))
    print(len(AllT))
 
    # create homogeneous transform matrices
    # these align each point cloud with the GMM model
    T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    # relative transform between 1 and 2
    # inverse brings 2nd point cloud back to common frame
    print(len(T_1))
    T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]

    # average calibration pairs 
    print(T_calib)
    T_final = transformPCDs.mean_transform(T_calib)
    print("Calibration Error: \n")
    print(T_final)
    gmmcalib_result = [T_final, X]
    with open("/app/output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump(gmmcalib_result, f) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration script")
    parser.add_argument("--data_path", type=str, help="Path to data", default="../data")
    parser.add_argument("--config_file_path", type=str, help="Path to config file", default="../config/config.yaml")
    parser.add_argument("--sequence", nargs='+', type=int, help="Sequence sequence of pcds")

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file_path))

    if args.sequence is None:
        sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    else:
        sequence = args.sequence

    calibrate(data_path, config_file_path, sequence)

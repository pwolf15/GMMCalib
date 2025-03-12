import argparse
import os
import generatePCDs
import transformPCDs
from modelgenerator import jgmm
from socket_client import SocketIOClient
import create_gt
import numpy as np
import pickle
import csv
from scipy.spatial.transform import Rotation as R

def get_initial_positions(V, nObs, num_sensors):

    initial_positions = {}
    initial_positions["sensors"] = list(range(1, num_sensors + 1))
    initial_positions["num_obs"] = nObs // num_sensors
    initial_positions["pcd_list"] = []
    for idx in range(0, nObs):
        sensor_id = idx // initial_positions["num_obs"] 
        obs_id = idx % initial_positions["num_obs"] 
        initial_positions["pcd_list"].append({
            "file": f'{sensor_id}_{obs_id}',
            "points": V[idx].tolist()
        })
    return initial_positions


def calibrate(data_path, config_file_path, sequence, num_iter=100, fixCentroids=False):
    pcds, num_sensors = generatePCDs.generate_data(data_path, config_file_path, sequence)
    Xin = create_gt.create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=400) + np.array([9.8, 4.75, 0.38])

    V = [np.array(cloud.points) for cloud in pcds]
    nObs = len(V)

    # create socket client to publish display data
    client = SocketIOClient('http://127.0.0.1:5000')
    # client = None

    # send initial point cloud positions (vehicle frame)
    initial_positions = get_initial_positions(V, nObs, num_sensors)
    if client:
        client.emit("initial_positions", initial_positions)

        # send initial gmm means
        client.emit("gmm_means", {"Xin": Xin.tolist(), "X": Xin.tolist(), "num_iter": 0})

    print("####### Perform Calibration and Model Generation. ########")
    X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=num_iter, socket_client=client, num_sensors=num_sensors, fixCentroids=fixCentroids)
 
    T_1 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    T_2 = [transformPCDs.homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
    T_final = transformPCDs.mean_transform(T_calib)
    print("Calibration Error: \n")
    print(T_final)
    gmmcalib_result = [T_final, X]
    with open("/app/output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump(gmmcalib_result, f) 

    return T_final

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
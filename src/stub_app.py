from socket_client import SocketIOClient
import numpy as np
import time

if __name__ == '__main__':

    # connect to Flask Socket.IO server
    client = SocketIOClient('http://127.0.0.1:5000')

    # 1. generate initial positions
    num_sensors = 5
    initial_positions = {}
    initial_positions["sensors"] = []
    initial_positions["num_obs"] = 1
    initial_positions["pcd_list"] = []
    for sensor in range(0, num_sensors):
        initial_positions["sensors"].append(str(sensor))
        points = np.random.rand(100, 3).tolist()
        initial_positions["pcd_list"].append({
            "file": f'{sensor}_1',
            "points": points
        })
    print(initial_positions)

    client.emit("initial_positions", initial_positions)

    # 2. generate initial GMM positions
    #   augment them over n frames
    num_iterations = 100
    k = 400
    Xin = np.random.rand(k, 3).tolist()
    X = Xin
    client.emit("gmm_means", {"Xin": Xin, "X": X, "num_iter": 0})
    time.sleep(1)

    for i in range(0, num_iterations):
        X = np.random.rand(k, 3).tolist()
        client.emit("gmm_means", {"Xin": Xin, "X": X, "num_iter": i+1})
        time.sleep(0.5)

    time.sleep(2)

    # # 1. send config
    # while True:
    #     # ✅ Generate random 3D point cloud
    #     points = np.random.rand(100, 3).tolist()
    #     print(f"📤 Sending Point Cloud Data: {points[:3]} ...")  # Show first 3 points for debug

    #     # ✅ Emit point cloud to the WebSocket server
    #     client.emit('point_cloud', {'points': points})

    #     time.sleep(1)  # ✅ Send updates every second
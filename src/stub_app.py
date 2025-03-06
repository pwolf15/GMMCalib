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

    time.sleep(2)

    # # 1. send config
    # while True:
    #     # ✅ Generate random 3D point cloud
    #     points = np.random.rand(100, 3).tolist()
    #     print(f"📤 Sending Point Cloud Data: {points[:3]} ...")  # Show first 3 points for debug

    #     # ✅ Emit point cloud to the WebSocket server
    #     client.emit('point_cloud', {'points': points})

    #     time.sleep(1)  # ✅ Send updates every second
import numpy as np
from modelgenerator import jgmm

from socket_client import SocketIOClient
import os

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

# --- synthetic data setup ---
plane1 = np.random.rand(3, 100)
plane2 = np.random.rand(3, 100) + np.array([[5], [0], [0]])
V = [plane1.T, plane2.T]      # jgmm will transpose internally back to 3×N
Xin = np.array([[0, 5],
                [0, 0],
                [0, 0]], dtype=float)

display = True
if display:
    client = SocketIOClient('http://127.0.0.1:5000')
else:
    client = None

nObs = 2
num_sensors = 2
initial_positions = get_initial_positions(V, nObs, num_sensors)

if client:
    client.emit("initial_positions", initial_positions)

    # send initial gmm means
    client.emit("gmm_means", {"Xin": Xin.T.tolist(), "X": Xin.T.tolist(), "num_iter": 0})

# Run jgmm for one EM iteration, exporting labels
X, TV, T, pk = jgmm(
    V, Xin.T,
    maxNumIter=10,
    num_sensors=2,
    fixCentroids=True
)

# After this you'll find:
#   labels_sensor_0.txt   # 100 lines, each an int in [0,1]
#   labels_sensor_1.txt   # 100 lines, each an int in [0,1]


# --- SMOKE TEST on the output ---
labels0 = np.loadtxt("labels_sensor_0.txt", dtype=int)
labels1 = np.loadtxt("labels_sensor_1.txt", dtype=int)

# 1) Correct number of labels
assert labels0.shape == (100,), f"Got {labels0.shape[0]} labels, expected 100"
assert labels1.shape == (100,), f"Got {labels1.shape[0]} labels, expected 100"

# 2) Labels in valid range [0, K-1]
K = Xin.shape[1]
assert labels0.min() >= 0 and labels0.max() < K, f"Labels0 out of range: {labels0.min()}..{labels0.max()}"
assert labels1.min() >= 0 and labels1.max() < K, f"Labels1 out of range: {labels1.min()}..{labels1.max()}"

print("Smoke test passed. Label distributions:")
print("  View-0 unique labels:", np.unique(labels0))
print("  View-1 unique labels:", np.unique(labels1))

# --- cleanup ---
os.remove("labels_sensor_0.txt")
os.remove("labels_sensor_1.txt")
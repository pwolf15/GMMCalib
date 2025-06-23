import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

# Load estimated transformation from file
with open("./output/gmmcalib_result.pkl", "rb") as f:
    T_final, _ = pickle.load(f)

print(T_final)

# Define ground truth (identity matrix)
T_ref = np.eye(4)

# Compute translation error
t_est = T_final[:3, 3]
t_ref = T_ref[:3, 3]
translation_error = np.linalg.norm(t_est - t_ref)

# Compute rotation error
R_est = T_final[:3, :3]
R_gt = np.eye(3)
R_err = R.from_matrix(R_gt.T @ R_est)
rotation_error_rad = R_err.magnitude()
rotation_error_deg = np.degrees(rotation_error_rad)

# Print result
print("Comparing estimated transform with identity matrix:")
print(f"Translation error: {translation_error:.6f} meters")
print(f"Rotation error: {rotation_error_rad:.6f} radians / {rotation_error_deg:.6f} degrees")

# Define tolerance
max_translation_error = 0.03  # 1 cm
max_rotation_error_rad = 0.004  # 0.1 degrees

# Threshold check
if translation_error > max_translation_error and rotation_error_rad > max_rotation_error_rad:
    print("Calibration is outside acceptable tolerance.")
    print(f"max translation error: {max_translation_error}")
    print(f"max rotation error (rad): {max_rotation_error_rad}")
    exit(1)
elif translation_error > max_translation_error:
    print(f"Translation error is outside acceptable tolerance. max translation error: {max_translation_error}")
    exit(1)
elif rotation_error_rad > max_rotation_error_rad:
    print(f"Translation error is outside acceptable tolerance. max rotation error (rad): {max_rotation_error_rad}")
    exit(1)
else:
    print("Calibration is within acceptable tolerance.")

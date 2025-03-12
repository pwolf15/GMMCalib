from gmmcalib_viz import calibrate 
import csv
from scipy.spatial.transform import Rotation as R
import yaml
import time

def update_use_noise(YAML_FILE, new_value):
    """Loads the YAML file, modifies 'use_noise', and saves it back with correct formatting."""
    
    # ✅ Load the YAML file
    with open(YAML_FILE, "r") as file:
        config = yaml.safe_load(file)

    # ✅ Update the 'use_noise' parameter
    config["use_noise"] = new_value

    # ✅ Save the updated YAML file while preserving formatting
    with open(YAML_FILE, "w") as file:
        yaml.dump(config, file, default_flow_style=None, sort_keys=False, allow_unicode=True)

    print(f"Updated 'use_noise' to {new_value} in {YAML_FILE}")


def analyze():

    
    obs_params = [10]
    iter_params = [100]
    noise_params = [0]
    fix_centroid_params = [0, 1]

    # obs_params = [10]
    # iter_params = [10]
    # noise_params = [0, 1]
    # fix_centroid_params = [0, 1]

    data_path = './point_clouds_overlap'
    config_file_path = './config/cube_config.yaml'
    columns = [
        'algorithm',
        'data',
        'config',
        'numIterations',
        'sequence',
        'sequenceLength',
        'useNoise',
        'fixCentroids',
        'eulerRollRad',
        'eulerPitchRad',
        'eulerYawRad',
        'deltaTx',
        'deltaTy',
        'deltaTz',
        'executionTimeSec'
    ]
    with open('./results/results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for obs_param in obs_params:
            for iter_param in iter_params:
                for noise_param in noise_params:
                    for fix_centroid_param in fix_centroid_params:

                        # set sequence
                        start = 121
                        sequence = list(range(start,start + obs_param))

                        # write noise param to file
                        update_use_noise(config_file_path, noise_param)

                        # run calibration
                        start_time = time.time()
                        T_final = calibrate(data_path, config_file_path, sequence, iter_param, fix_centroid_param)
                        end_time = time.time() 
                        execution_time = end_time - start_time 
                        
                        # Extract rotation matrix (top-left 3x3)
                        rotation_matrix = T_final[:3, :3]

                        # Convert rotation matrix to Euler angles (degrees)
                        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)  # Roll, Pitch, Yaw

                        # Extract translation components (X, Y, Z)
                        translation_vector = T_final[:3, 3]
                        x_translation, y_translation, z_translation = translation_vector

                        # Prepare data for logging (6 values)
                        # algorithm, dataset, config file, num iterations, sequence, with or without noise, fix centroids, data
                        new_data = ['GMM', data_path, config_file_path, iter_param, sequence, len(sequence), noise_param, fix_centroid_param, euler_angles[0], euler_angles[1], euler_angles[2], x_translation, y_translation, z_translation, str(execution_time)]

                        # Append data to CSV file
                        writer = csv.writer(file)
                        writer.writerow(new_data)
                        file.flush()

if __name__ == "__main__":
    analyze()
    
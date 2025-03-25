from gmmcalib_viz import calibrate 
import csv
from scipy.spatial.transform import Rotation as R
import yaml
import time
import tempfile
import math

# hack to dynamically set use_noise parameter in GMMCalib config yaml
def update_config_param(input_yaml, output_yaml, field, new_value):
    """Loads the YAML file, modifies 'use_noise', and saves it back with correct formatting."""
    
    # load yaml
    with open(input_yaml, "r") as file:
        config = yaml.safe_load(file)

    # set the 'use_noise' parameter
    config[field] = new_value

    # save updated YAML file while preserving formatting
    with open(output_yaml, "w") as file:
        yaml.dump(config, file, default_flow_style=None, sort_keys=False, allow_unicode=True)

    print(f"Updated {field} to {new_value} in {output_yaml}")

def analyze():

    ## parameters for consecutive runs of GMMCalib

    obs_params = [10]
    iter_params = [100]
    noise_params = [1]
    fix_centroid_params = [0]
    rotation_error_params = [
        [0,0,0]
    ]
    translation_error_params = [
        [0,0,0]
    ]

    # sequence length
    # obs_params = [100]

    # # num GMM EM iterations
    # iter_params = [100]

    # # use data with noise (1) or not (0)
    # noise_params = [0]

    # # fix GMM means (1) or not (0)
    # fix_centroid_params = [0]

    data_path = './point_clouds_overlap'
    config_file_path = './config/cube_config.yaml'
    results_file_path = './results/results.csv'

    ## end parameters

    # CSV results file columns
    columns = [
        'algorithm',
        'data',
        'config',
        'numIterations',
        'sequence',
        'sequenceLength',
        'useNoise',
        'fixCentroids',
        'predRollRad',
        'trueRollRad',
        'predPitchRad',
        'truePitchRad',
        'predYawRad',
        'trueYawRad',
        'predTx',
        'trueTx',
        'predTy',
        'trueTy',
        'predTz',
        'trueTz',
        'diffRollRad',
        'diffPitchRad',
        'diffYawRad',
        'diffTx',
        'diffTy',
        'diffTz',
        'executionTimeSec'
    ]
    with open(results_file_path, 'w', newline='') as file, tempfile.NamedTemporaryFile() as tmpfile:

        writer = csv.writer(file)
        writer.writerow(columns)

        # iterate through parameters
        # run GMM calib ('calibrate') for each parameter combination
        for obs_param in obs_params:
            for iter_param in iter_params:
                for noise_param in noise_params:
                    for fix_centroid_param in fix_centroid_params:
                        for rotation_error_param in rotation_error_params:
                            for translation_error_param in translation_error_params:

                                # set sequence

                                # TODO fix hard-coding of start sequence
                                start = 121
                                sequence = list(range(start,start + obs_param))

                                # write noise param to file
                                update_config_param(config_file_path, tmpfile.name, "use_noise", noise_param)
                                update_config_param(tmpfile.name, tmpfile.name, "rotation_error", [rotation_error_param])
                                update_config_param(tmpfile.name, tmpfile.name, "translation_error", [translation_error_param])

                                # run calibration, record execution time
                                start_time = time.time()
                                T_final = calibrate(data_path, tmpfile.name, sequence, iter_param, fix_centroid_param)
                                end_time = time.time() 
                                execution_time = end_time - start_time 
                                
                                ## write out rotation (euler angles: roll, pitch, yaw), and translation (tx, ty, tz)
                                # Extract rotation matrix (top-left 3x3)
                                rotation_matrix = T_final[:3, :3]

                                # Convert rotation matrix to Euler angles (degrees)
                                euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)  # Roll, Pitch, Yaw

                                # Extract translation components (X, Y, Z)
                                translation_vector = T_final[:3, 3]
                                x_translation, y_translation, z_translation = translation_vector

                                # Prepare data for logging
                                # algorithm, dataset, config file, num iterations, sequence, with or without noise, fix centroids, data
                                new_data = [
                                    'GMM', 
                                    data_path, 
                                    config_file_path, 
                                    iter_param, 
                                    sequence, 
                                    len(sequence), 
                                    noise_param, 
                                    fix_centroid_param, 
                                    euler_angles[0], 
                                    math.radians(rotation_error_param[0]),
                                    euler_angles[1], 
                                    math.radians(rotation_error_param[1]),
                                    euler_angles[2], 
                                    math.radians(rotation_error_param[2]),
                                    x_translation, 
                                    translation_error_param[0],
                                    y_translation, 
                                    translation_error_param[1],
                                    z_translation, 
                                    translation_error_param[2],
                                    euler_angles[0] - math.radians(rotation_error_param[0]),
                                    euler_angles[1] - math.radians(rotation_error_param[1]),
                                    euler_angles[2] - math.radians(rotation_error_param[2]),
                                    x_translation - translation_error_param[0],
                                    y_translation - translation_error_param[1],
                                    z_translation - translation_error_param[2],
                                    str(execution_time)]

                                # Append data to CSV file
                                writer = csv.writer(file)
                                writer.writerow(new_data)
                                file.flush()

if __name__ == "__main__":
    analyze()
    
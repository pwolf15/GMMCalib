from gmmcalib_viz import calibrate 
import csv
from scipy.spatial.transform import Rotation as R
import yaml
import time
import tempfile
import math
import numpy as np

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

    sim_iterations = 1
    obs_params = [1]
    iter_params = [50]
    noise_params = [0]
    fix_centroid_params = [0]
    rotation_error_params = [
        [0,0,0]
    ]
    translation_error_params = [
        [0,0,0]
    ]
    num_points_params = [
        # 50,
        100,
        # 100,
        # 200,
        # 400
    ]

    # sequence length
    # obs_params = [100]

    # # num GMM EM iterations
    # iter_params = [100]

    # # use data with noise (1) or not (0)
    # noise_params = [0]

    # # fix GMM means (1) or not (0)
    # fix_centroid_params = [0]

    config_type = 'cube'
    # config_type = 'default'

    if config_type == 'cube':
        data_path = './point_clouds_overlap'
        config_file_path = './config/cube_config.yaml'
        results_file_path = './results/cube_fix_centroids_results.csv'
        start = 121
    else:
        data_path = './data'
        config_file_path = './config/config.yaml'
        results_file_path = './results/default_results.csv'
        start = 1
        obs_params = [3]

    ## end parameters

    # CSV results file columns
    columns = [
        'simIter',
        'algorithm',
        'data',
        'config',
        'numIterations',
        'sequence',
        'sequenceLength',
        'useNoise',
        'fixCentroids',
        'numPoints',
        'trueRollRad',
        'truePitchRad',
        'trueYawRad',
        'trueTx',
        'trueTy',
        'trueTz',
        'predRollRad',
        'predPitchRad',
        'predYawRad',
        'predTx',
        'predTy',
        'predTz',
        'diffRollRad',
        'diffPitchRad',
        'diffYawRad',
        'diffTx',
        'diffTy',
        'diffTz',
        'executionTimeSec'
    ]

    # iterate through parameters
    # run GMM calib ('calibrate') for each parameter combination
    for obs_param in obs_params:
        for iter_param in iter_params:
            for noise_param in noise_params:
                for fix_centroid_param in fix_centroid_params:
                    for num_points_param in num_points_params:

                        results_file_path = f'results/results_{config_type}_noise_{noise_param}_fixcentroids_{fix_centroid_param}_numobs_{obs_param}.csv'
                        with open(results_file_path, 'w', newline='') as file, tempfile.NamedTemporaryFile() as tmpfile:

                            writer = csv.writer(file)
                            writer.writerow(columns)
                            for sim_iter in range(0,sim_iterations):
                                # set sequence
                                max_rot_deg=3.0  # degrees
                                max_trans=0.1   # meters
                                np.random.seed(None)
                                angle_deg = np.random.uniform(-max_rot_deg, max_rot_deg, size=3)
                                translation = np.random.uniform(-max_trans, max_trans, size=3)
                                rotation_error_param = rotation_error_params[0] # angle_deg.tolist()
                                translation_error_param = translation_error_params[0] # translation.tolist()
                                print('injected errors', rotation_error_param, translation_error_param)
                                
                                # TODO fix hard-coding of start sequence
                                sequence = list(range(start,start + obs_param))

                                # write noise param to file
                                update_config_param(config_file_path, tmpfile.name, "use_noise", noise_param)
                                update_config_param(tmpfile.name, tmpfile.name, "rotation_error", [rotation_error_param])
                                update_config_param(tmpfile.name, tmpfile.name, "translation_error", [translation_error_param])

                                # run calibration, record execution time
                                start_time = time.time()
                                T_final = calibrate(data_path, tmpfile.name, sequence, iter_param, fix_centroid_param, num_points_param)
                                end_time = time.time() 
                                execution_time = end_time - start_time 
                                
                                ## write out rotation (euler angles: roll, pitch, yaw), and translation (tx, ty, tz)
                                # Extract rotation matrix (top-left 3x3)
                                rotation_matrix = T_final[:3, :3]

                                # Convert rotation matrix to Euler angles (degrees)
                                euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)  # Roll, Pitch, Yaw
                                euler_angles_deg = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)  # Roll, Pitch, Yaw

                                # Extract translation components (X, Y, Z)
                                translation_vector = T_final[:3, 3]
                                x_translation, y_translation, z_translation = translation_vector

                                print('euler angles', euler_angles_deg)
                                print('translation', translation_vector)
                                print('angle error', rotation_error_param)
                                print('translation error', translation_error_param)
                            
                                errors = [
                                    euler_angles[0] - math.radians(rotation_error_param[0]),
                                    euler_angles[1] - math.radians(rotation_error_param[1]),
                                    euler_angles[2] - math.radians(rotation_error_param[2]),
                                    x_translation - translation_error_param[0],
                                    y_translation - translation_error_param[1],
                                    z_translation - translation_error_param[2],
                                ]
                                # Prepare data for logging
                                # algorithm, dataset, config file, num iterations, sequence, with or without noise, fix centroids, data
                                new_data = [
                                    sim_iter,
                                    'GMM', 
                                    data_path, 
                                    config_file_path, 
                                    iter_param, 
                                    sequence, 
                                    len(sequence), 
                                    noise_param, 
                                    fix_centroid_param, 
                                    num_points_param,
                                    math.radians(rotation_error_param[0]),
                                    math.radians(rotation_error_param[1]),
                                    math.radians(rotation_error_param[2]),
                                    translation_error_param[0],
                                    translation_error_param[1],
                                    translation_error_param[2],
                                    euler_angles[0], 
                                    euler_angles[1],
                                    euler_angles[2], 
                                    x_translation, 
                                    y_translation, 
                                    z_translation, 
                                    errors[0],
                                    errors[1],
                                    errors[2],
                                    errors[3],
                                    errors[4],
                                    errors[5],
                                    str(execution_time)]

                                # from GMM calib paper: I estimate this
                                thresholds = [
                                    0.005, # 0.0033
                                    0.005, # 0.0036
                                    0.005, # 0.0020
                                    0.03,  # 0.015
                                    0.03,  # 0.027
                                    0.03,  # 0.018
                                ]

                                import logging
                                LOG_LEVEL = logging.DEBUG
                                LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
                                from colorlog import ColoredFormatter
                                logging.root.setLevel(LOG_LEVEL)
                                formatter = ColoredFormatter(LOGFORMAT)
                                stream = logging.StreamHandler()
                                stream.setLevel(LOG_LEVEL)
                                stream.setFormatter(formatter)
                                log = logging.getLogger('pythonConfig')
                                log.setLevel(LOG_LEVEL)
                                log.addHandler(stream)

                                labels = ['roll', 'pitch', 'yaw', 'x', 'y', 'z']

                                for idx in range(len(errors)):
                                    if abs(errors[idx]) > thresholds[idx]:
                                        log.error(f'{labels[idx]}: {errors[idx]:.3f} > {thresholds[idx]}')
                                    else:
                                        log.info(f'{labels[idx]}: {errors[idx]:.3f} <= {thresholds[idx]}')

                                # Append data to CSV file
                                writer = csv.writer(file)
                                writer.writerow(new_data)
                                file.flush()

if __name__ == "__main__":
    analyze()
    
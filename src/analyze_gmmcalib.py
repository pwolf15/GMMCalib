from gmmcalib_viz import calibrate 
import csv
from scipy.spatial.transform import Rotation as R
import yaml
import time
import tempfile
import math
import numpy as np


import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_dashboard_all_views_2x4_with_text(
    initial_data,
    final_data,
    config_params,
    output_error,
    output_path="output/dashboard_annotated.png",
    width=2400,
    height=1200,
    scale=2
):
    def build_trace(pcd_list, num_obs):
        traces = []
        colors = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "purple", 5: "cyan"}
        for idx, pcd in enumerate(pcd_list):
            sensor_id = idx // num_obs
            pts = np.array(pcd["points"])
            color = colors.get(sensor_id, "gray")
            trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2, color=color),
                name=f"Sensor {sensor_id+1}",
                showlegend=False
            )
            traces.append(trace)
        return traces

    init_traces = build_trace(initial_data["pcd_list"], initial_data["num_obs"])
    final_traces = build_trace(final_data["pcd_list"], final_data["num_obs"])

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{'type': 'scene'}]*4, [{'type': 'scene'}]*4],
        subplot_titles=[
            "Initial (Top)", "Initial (Side)", "Initial (Front)", "Initial (Iso)",
            "Final (Top)", "Final (Side)", "Final (Front)", "Final (Iso)"
        ]
    )

    cameras = {
        "top": dict(eye=dict(x=0.001, y=0.001, z=2.5)),
        "side": dict(eye=dict(x=2.5, y=0.001, z=0.001)),
        "front": dict(eye=dict(x=0.001, y=2.5, z=0.001)),
        "iso": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    }
    views = ["top", "side", "front", "iso"]

    for i, view in enumerate(views):
        for trace in init_traces:
            fig.add_trace(trace, row=1, col=i+1)
        for trace in final_traces:
            fig.add_trace(trace, row=2, col=i+1)
        fig.update_scenes(camera=cameras[view], row=1, col=i+1)
        fig.update_scenes(camera=cameras[view], row=2, col=i+1)

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=100, b=20),
    )

    # Add annotations for parameters and errors
    y_offset = 1.15
    lines = []
    lines.append("CONFIG:")
    for k, v in config_params.items():
        lines.append(f"{k}: {v}")
    lines.append("\nOUTPUT ERROR (deg, m):")
    for k, v in output_error.items():
        lines.append(f"{k}: {v:.3f}")

    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper", yref="paper",
        x=0, y=0.5,
        showarrow=False,
        align="left",
        font=dict(size=14),
        bordercolor="black",
        borderwidth=1
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, scale=scale)
    print(f"Saved 2x4 annotated dashboard to: {output_path}")

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
 
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)

    sim_iterations = params["sim_iterations"]
    obs_params = params["obs_params"]
    iter_params = params["iter_params"]
    noise_params = params["noise_params"]
    fix_centroid_params = params["fix_centroid_params"]
    rotation_error_params = params["rotation_error_params"]
    translation_error_params = params["translation_error_params"]
    num_points_params = params["num_points_params"]
    config_type = params["config_type"]
    inject_error = params["inject_error"]

    data_path = params["paths"][config_type]["data"]
    config_file_path = params["paths"][config_type]["config"]
    start = params["paths"][config_type]["start"]


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
                                if not inject_error:
                                    rotation_error_param = rotation_error_params[0]  # angle_deg.tolist() #
                                    translation_error_param = translation_error_params[0]
                                else:
                                    rotation_error_param = angle_deg.tolist() #
                                    translation_error_param = translation.tolist() #translation_error_params[0] # 
                                print('injected errors', rotation_error_param, translation_error_param)
                                
                                # TODO fix hard-coding of start sequence
                                sequence = list(range(start,start + obs_param))

                                # write noise param to file
                                update_config_param(config_file_path, tmpfile.name, "use_noise", noise_param)
                                update_config_param(tmpfile.name, tmpfile.name, "rotation_error", [rotation_error_param])
                                update_config_param(tmpfile.name, tmpfile.name, "translation_error", [translation_error_param])

                                # run calibration, record execution time
                                start_time = time.time()
                                T_final, initial_positions, final_registrations = calibrate(data_path, tmpfile.name, sequence, iter_param, fix_centroid_param, num_points_param)
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

                                output_errors = errors
                                save_dashboard_all_views_2x4_with_text(
                                    initial_data=initial_positions,
                                    final_data=final_registrations,
                                    config_params={
                                        "Noise": noise_param,
                                        "FixCentroids": fix_centroid_param,
                                        "NumObservations": obs_param,
                                        "NumPoints": num_points_param,
                                        "Iterations": iter_param
                                    },
                                    output_error={
                                        "Roll": output_errors[0],
                                        "Pitch": output_errors[1],
                                        "Yaw": output_errors[2],
                                        "X": output_errors[3],
                                        "Y": output_errors[4],
                                        "Z": output_errors[5],
                                    }
                                )

                                # Append data to CSV file
                                writer = csv.writer(file)
                                writer.writerow(new_data)
                                file.flush()

if __name__ == "__main__":
    analyze()
    
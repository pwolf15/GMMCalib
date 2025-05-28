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

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def pose_error(R_pred, t_pred, R_gt, t_gt):
    """
    Returns (angle_error_rad, translation_error_m)
    Angle = geodesic distance on SO(3)
    Translation = Euclidean distance after putting both poses in same frame
    """
    R_err = R_gt.T @ R_pred                       # relative rotation
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))

    t_err = R_gt.T @ (t_pred - t_gt)              # bring to gt frame
    trans = np.linalg.norm(t_err)
    return angle, trans

def save_dashboard_all_views_2x5_with_text(
    initial_data,
    final_data,
    config_params,
    output_error,
    Xin,
    X,
    output_path="output/dashboard_annotated.png",
    width=3000,
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
        rows=2, cols=5,
        specs=[[{'type': 'scene'}]*5, [{'type': 'scene'}]*5],
        subplot_titles=[
            "Initial (Top)", "Initial (Side)", "Initial (Front)", "Initial (Iso)", "Initial GMM (Iso)",
            "Final (Top)", "Final (Side)", "Final (Front)", "Final (Iso)", "Final GMM (Iso)"
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
            fig.update_scenes(
                camera=cameras[view],
                aspectmode="data",
                row=1, col=i+1
            )
            fig.update_scenes(
                camera=cameras[view],
                aspectmode="data",
                row=2, col=i+1
            )


    # GMM Xin (initial GMM means, iso view, top row, col 5)
    fig.add_trace(go.Scatter3d(
        x=Xin[:, 0], y=Xin[:, 1], z=Xin[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='Xin',
        showlegend=False
    ), row=1, col=5)
    fig.update_scenes(camera=cameras["iso"], aspectmode="data",row=1, col=5)

    # GMM X (final GMM means, iso view, bottom row, col 5)
    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode='markers',
        marker=dict(size=3, color='orange'),
        name='X',
        showlegend=False
    ), row=2, col=5)
    fig.update_scenes(camera=cameras["iso"], aspectmode="data", row=2, col=5)

    # Annotations for config + output error
    lines = ["CONFIG:"]
    for k, v in config_params.items():
        lines.append(f"{k}: {v}")
    lines.append("OUTPUT ERROR (deg, m):")
    for k, v in output_error.items():
        lines.append(f"{k}: {v:.3f}")

    fig.add_annotation(
        text="<br>".join(lines),
        xref="paper", yref="paper",
        x=0.0, y=0.5,
        showarrow=False,
        align="left",
        font=dict(size=14),
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=100, b=20),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path, scale=scale)
    print(f"✅ Saved dashboard to: {output_path}")


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
                                T_final, initial_positions, final_registrations, Xin, X = calibrate(data_path, tmpfile.name, sequence, iter_param, fix_centroid_param, num_points_param)
                                end_time = time.time() 
                                execution_time = end_time - start_time 
                                
                                # ------------------------------------------------------------------
                                # ❶  Ground-truth pose you injected (rotation_error_param is in deg)
                                R_gt = R.from_euler('xyz',
                                                    rotation_error_param,       # list in degrees
                                                    degrees=True).as_matrix()    # (3,3)
                                t_gt = np.asarray(translation_error_param)       # (3,)

                                # ❷  Predicted pose returned by calibrate()
                                R_pred = T_final[:3, :3]                         # (3,3)
                                t_pred = T_final[:3, 3]                          # (3,)

                                # ❸  Pose error in the *same* reference frame
                                R_err  = R_gt.T @ R_pred                         # relative rotation
                                t_err  = R_gt.T @ (t_pred - t_gt)                # relative translation
                                print("pose error (deg, cm):",
                                    np.degrees(np.linalg.norm(R.from_matrix(R_err).as_rotvec())),
                                    np.linalg.norm(t_err)*100)
                                # invert predicted pose and see if error plummets
                                R_pred_inv = R_pred.T
                                t_pred_inv = -R_pred.T @ t_pred

                                ang1, trans1 = pose_error(R_pred,      t_pred,      R_gt, t_gt)
                                ang2, trans2 = pose_error(R_pred_inv,  t_pred_inv,  R_gt, t_gt)
                                print("error direct :", np.degrees(ang1), trans1*100, "cm")
                                print("error inverted:", np.degrees(ang2), trans2*100, "cm")

                                # ❹  Convert to roll / pitch / yaw (rad)  &  xyz (m)
                                roll_err, pitch_err, yaw_err = R.from_matrix(R_err).as_euler('xyz', degrees=False)
                                x_err,    y_err,    z_err    = t_err

                                errors = [roll_err, pitch_err, yaw_err, x_err, y_err, z_err]
                                # ------------------------------------------------------------------

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
                                    roll_err,
                                    pitch_err,
                                    yaw_err,
                                    t_pred[0], 
                                    t_pred[1], 
                                    t_pred[2], 
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
                                save_dashboard_all_views_2x5_with_text(
                                    initial_data=initial_positions,
                                    final_data=final_registrations,
                                    Xin=Xin,
                                    X=X,
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
    
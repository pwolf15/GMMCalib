import numpy as np
import open3d as o3d
import pickle
from transformPCDs import compute_global_transform
import yaml


def generate_data(data_path, config_file_path, sequence):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_sensor_1 = config_data.get("transform_sensor_1", "")[0]
    transform_sensor_2 = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]
    number_of_sensors = config_data.get("number_of_sensors", "")

    # cube data additions
    is_cube_data = config_data.get("is_cube_data", False)
    use_noise = config_data.get("use_noise", False)
    # file_indices = config_data.get("file_indices", [[]])

    print(f'is cube data?: {is_cube_data}')
    print(f'use noise?: {use_noise}')
    print(f'seqeuence?: {sequence}')

    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    
    if not is_cube_data:
        for sensor in sensors:
            for frame in range(sequence[0], sequence[-1]+1):
                pcd_raw = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")
                pcd = pcd_raw
                # carla transform
                iso_points = np.asarray(pcd_raw.points)
                iso_points[:,:2] = iso_points[:,:2]*-1
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(iso_points)

                if sensor == sensors[0]:
                    T_g = compute_global_transform(transform_sensor_1[3:], transform_sensor_1[:3])
                else: 
                    T_g = compute_global_transform(transform_sensor_2[3:], transform_sensor_2[:3])

                pcd.transform(T_g)
                # Crop 
                roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                pcds.append(pcd.crop(roi))
    else:
        for sensor in sensors:
            sensor_id = int(sensor[-2])
            prefix = 'frontleft' if sensor_id == 1 else 'frontright'
            prefix += 'withnoise' if use_noise else 'nonoise'
            for idx in range(sequence[0], sequence[-1] + 1):
                filename = f'{data_path}/{idx}_{prefix}.pcd'
                pcd_raw = o3d.io.read_point_cloud(filename)
                pcd = pcd_raw
                # carla transform
                iso_points = np.asarray(pcd_raw.points)
                iso_points[:,:2] = iso_points[:,:2]*-1
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(iso_points)

                if sensor == sensors[0]:
                    T_g = compute_global_transform(transform_sensor_1[3:], transform_sensor_1[:3])
                else: 
                    T_g = compute_global_transform(transform_sensor_2[3:], transform_sensor_2[:3])

                pcd.transform(T_g)
                # Crop 
                roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                pcds.append(pcd.crop(roi))
                

    return pcds

def generate_cube_data(data_path, config_file_path):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_sensor_1 = config_data.get("transform_sensor_1", "")[0]
    transform_sensor_2 = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]
    number_of_sensors = config_data.get("number_of_sensors", "")
    
    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    for sensor in sensors:
        for frame in range(sequence[0], sequence[-1]+1):
            pcd_raw = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")
            pcd = pcd_raw
            # carla transform
            iso_points = np.asarray(pcd_raw.points)
            iso_points[:,:2] = iso_points[:,:2]*-1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(iso_points)

            if sensor == sensors[0]:
                T_g = compute_global_transform(transform_sensor_1[3:], transform_sensor_1[:3])
            else: 
                T_g = compute_global_transform(transform_sensor_2[3:], transform_sensor_2[:3])

            pcd.transform(T_g)
            # Crop 
            roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcds.append(pcd.crop(roi))
    return pcds
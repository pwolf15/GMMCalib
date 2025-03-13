import numpy as np
import open3d as o3d
import pickle
from transformPCDs import compute_global_transform
import yaml


def generate_data(data_path, config_file_path, sequence):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    number_of_sensors = config_data.get("number_of_sensors", 2)  # Default to 2 if not provided
    transform_sensors = [
        config_data.get(f"transform_sensor_{i+1}", [0,0,0,0,0,0])[0]
        for i in range(number_of_sensors)
    ]
    print(transform_sensors)

    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]

    # cube data additions
    is_cube_data = config_data.get("is_cube_data", False)
    use_noise = config_data.get("use_noise", False)
    use_carla_transform = config_data.get("use_carla_transform", False)

    print(f'is cube data?: {is_cube_data}')
    print(f'use noise?: {use_noise}')
    print(f'seqeuence?: {sequence}')

    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    
    if not is_cube_data:
        idx = 0
        for sensor in sensors:
            for frame in range(sequence[0], sequence[-1]+1):
                pcd_raw = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")
                pcd = pcd_raw

                # carla transform
                if use_carla_transform:
                    iso_points = np.asarray(pcd_raw.points)
                    iso_points[:,:2] = iso_points[:,:2]*-1
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(iso_points)

                if sensor == sensors[0]:
                    T_g = compute_global_transform(transform_sensors[0][3:], transform_sensors[0][:3])
                else: 
                    T_g = compute_global_transform(transform_sensors[idx][3:], transform_sensors[idx][:3])

                pcd.transform(T_g)
                # Crop 
                roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                print(pcd.crop(roi))
                pcds.append(pcd.crop(roi))
            idx += 1
    else:
        sensor_idx = 0
        skipped_indices = []
        for sensor in sensors:
            sensor_id = int(sensor[-2])
            prefix = 'frontleft' if sensor_id == 1 else 'frontright'
            prefix += 'withnoise001' if use_noise else 'nonoise'
            for idx in range(sequence[0], sequence[-1] + 1):

                filename = f'{data_path}/{idx}_{prefix}.pcd'
                pcd_raw = o3d.io.read_point_cloud(filename)
                pcd = pcd_raw
                print(filename)

                if len(pcd_raw.points) == 0:
                    print(f'warning: {filename} has 0 points')
                    skipped_indices.append(idx)

            sensor_idx += 1

        print('skipped indices: ', skipped_indices)

        sensor_idx = 0
        for sensor in sensors:
            sensor_id = int(sensor[-2])
            prefix = 'frontleft' if sensor_id == 1 else 'frontright'
            prefix += 'withnoise001' if use_noise else 'nonoise'

            # check empty
            for idx in range(sequence[0], sequence[-1] + 1):

                if idx in skipped_indices:
                    continue

                filename = f'{data_path}/{idx}_{prefix}.pcd'
                pcd_raw = o3d.io.read_point_cloud(filename)
                pcd = pcd_raw
                print(filename)

                if len(pcd_raw.points) == 0:
                    print(f'warning: {filename} has 0 points')


                # carla transform
                if use_carla_transform:
                    iso_points = np.asarray(pcd_raw.points)
                    iso_points[:,:2] = iso_points[:,:2]*-1
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(iso_points)

                if sensor == sensors[0]:
                    T_g = compute_global_transform(transform_sensors[sensor_idx][3:], transform_sensors[sensor_idx][:3])
                else: 
                    T_g = compute_global_transform(transform_sensors[sensor_idx][3:], transform_sensors[sensor_idx][:3])

                pcd.transform(T_g)
                # Crop 
                roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                print(pcd.crop(roi))
                pcds.append(pcd.crop(roi))

            sensor_idx += 1
        # exit(1)
                

    return pcds, number_of_sensors
import os

import pyvista as pv
import cv2
import numpy as np


def segment_bounding_box(
    point_cloud: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
):
    x_idcs = np.logical_and(point_cloud[:, 0] >= x_min, point_cloud[:, 0] < x_max)
    y_idcs = np.logical_and(point_cloud[:, 1] >= y_min, point_cloud[:, 1] < y_max)
    z_idcs = np.logical_and(point_cloud[:, 2] >= z_min, point_cloud[:, 2] < z_max)
    idcs = np.logical_and(np.logical_and(x_idcs, y_idcs), z_idcs)
    return point_cloud[idcs]


def clean_data(x: np.ndarray, y: np.ndarray, z: np.ndarray, rgb: np.ndarray):
    idcs = np.isfinite(x)
    x = x[idcs]
    y = y[idcs]
    z = z[idcs]
    rgb = rgb[idcs]
    return x, y, z, rgb


def load_points(
    point_cloud_prefix: str, x_path: str, y_path: str, z_path: str, rgba_path: str
):
    x = np.load(os.path.join(point_cloud_prefix, x_path))
    y = np.load(os.path.join(point_cloud_prefix, y_path))
    z = np.load(os.path.join(point_cloud_prefix, z_path))
    rgba = np.load(os.path.join(point_cloud_prefix, rgba_path))
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    x, y, z, rgb = clean_data(x, y, z, rgb)
    return x, y, z, rgb


def visual_inspection(
    stereo_cloud: np.ndarray,
    rgb: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    plotter = pv.Plotter()
    cloud = pv.PolyData(stereo_cloud)
    box = pv.Box(
        bounds=(x_min, x_max, y_min, y_max, z_min, z_max),
    )
    plotter.add_points(cloud, scalars=rgb, rgb=True)
    plotter.add_mesh(box, color="red")
    plotter.add_axes()
    plotter.show()


def main() -> None:
    # prefix = "/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/faros_integration_week_kuka_right"
    prefix = "/media/martin/Samsung_T5/23_07_04_faros_integration_week_measurements/self_registration/self_observation_rosbag"
    point_cloud_prefix = f"{prefix}/point_cloud"
    idx = 1200

    # process point cloud
    x, y, z, rgb = load_points(
        point_cloud_prefix,
        f"x_{idx}.npy",
        f"y_{idx}.npy",
        f"z_{idx}.npy",
        f"rgba_{idx}.npy",
    )

    stereo_cloud = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    left_params = {
        "x_min": 1.0,
        "x_max": 1.5,
        "y_min": -0.5,
        "y_max": 0.5,
        "z_min": -0.34,
        "z_max": 0.5,
    }

    right_params = {
        "x_min": 0.7,
        "x_max": 1.35,
        "y_min": -1.4,
        "y_max": -0.4,
        "z_min": -0.34,
        "z_max": 0.5,
    }

    visual_inspection(stereo_cloud, rgb.reshape(-1, 3), **right_params)

    stereo_cloud = segment_bounding_box(stereo_cloud, **right_params)
    plotter = pv.Plotter()
    cloud = pv.PolyData(stereo_cloud)
    plotter.add_points(cloud)
    plotter.show()


if __name__ == "__main__":
    main()

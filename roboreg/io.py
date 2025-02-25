import os
import pathlib
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rich
import yaml
from pytorch_kinematics import urdf_parser_py
from torch.utils.data import Dataset


class URDFParser:
    __slots__ = ["_urdf", "_robot"]

    def __init__(self) -> None:
        self._urdf = None
        self._robot = None

    def from_urdf(self, urdf: str) -> None:
        r"""Instantiate URDF parser from URDF string.

        Args:
            urdf (str): URDF string.
        """
        self._urdf = urdf
        self._robot = urdf_parser_py.urdf.Robot.from_xml_string(urdf)

    def from_ros_xacro(self, ros_package: str, xacro_path: str) -> None:
        r"""Instantiate URDF parser from ROS xacro file.

        Args:
            ros_package (str): Internally finds the path to ros_package.
            xacro_path (str): Path to xacro file relative to ros_package.
        """
        self.from_urdf(
            urdf=self.urdf_from_ros_xacro(
                ros_package=ros_package, xacro_path=xacro_path
            )
        )

    def urdf_from_ros_xacro(self, ros_package: str, xacro_path: str) -> str:
        r"""Convert ROS xacro file to URDF.

        Args:
            ros_package (str): Internally finds the path to ros_package.
            xacro_path (str): Path to xacro file relative to ros_package.

        Returns:
            str: URDF string.
        """

        import xacro
        from ament_index_python import get_package_share_directory

        self._urdf = xacro.process(
            os.path.join(get_package_share_directory(ros_package), xacro_path)
        )
        return self._urdf

    def chain_link_names(self, root_link_name: str, end_link_name: str) -> List[str]:
        r"""Get link names in chain from root to end link.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.

        Returns:
            List[str]: List of link names in chain from root_link_name to end_link_name.
        """
        self._verify_links_in_chain(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        link_names = [root_link_name]
        while link_names[-1] != end_link_name:
            children = self._robot.child_map[link_names[-1]]
            if len(children) != 1:
                raise RuntimeError(f"Expected 1 child, got {len(children)}.")
            _, child_link_name = children[0]
            if link_names[-1] == child_link_name:
                raise RuntimeError(f"End of chain without {end_link_name}.")
            link_names.append(child_link_name)
        return link_names

    def link_names_with_meshes(self, collision: bool = False) -> List[str]:
        r"""Get link names that have meshes.

        Args:
            collision (bool): If True, get collision meshes, else visual meshes.

        Returns:
            List[str]: List of link names with meshes.
        """
        links = [link.name for link in self._robot.links]
        for link in links:
            if collision:
                if not self._robot.link_map[link].collision:
                    links.remove(link)
            else:
                if not self._robot.link_map[link].visual:
                    links.remove(link)
        return links

    def raw_mesh_paths(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, str]:
        r"""Get the raw mesh paths as specified in URDF.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh paths, else visual mesh paths.

        Returns:
            Dict[str,str]: Dictionary of link names and raw mesh paths.
        """
        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        raw_mesh_paths = {}
        # lookup paths
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if collision:
                if link.collision is None:
                    continue
                raw_mesh_paths[link_name] = link.collision.geometry.filename
            else:
                if link.visual is None:
                    continue
                raw_mesh_paths[link_name] = link.visual.geometry.filename
        return raw_mesh_paths

    def ros_package_mesh_paths(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, str]:
        r"""Get the absolute mesh paths by resolving package within ROS.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh paths, else visual mesh paths.

        Returns:
            Dict[str,str]: Dictionary of link names and absolute mesh paths.
        """
        raw_mesh_paths = self.raw_mesh_paths(
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            collision=collision,
        )
        from ament_index_python import get_package_share_directory

        ros_package_mesh_paths = {}
        for link_name in raw_mesh_paths.keys():
            raw_mesh_path = raw_mesh_paths[link_name]
            if raw_mesh_path.startswith("package://"):
                raw_mesh_path = raw_mesh_path.replace("package://", "")
                package, relative_mesh_path = raw_mesh_path.split("/", 1)
                ros_package_mesh_paths[link_name] = os.path.join(
                    get_package_share_directory(package), relative_mesh_path
                )
            else:
                raise ValueError("Case unhandled.")
        return ros_package_mesh_paths

    def mesh_origins(
        self, root_link_name: str, end_link_name: str, collision: bool = False
    ) -> Dict[str, np.ndarray]:
        r"""Get mesh origins.

        Args:
            root_link_name (str): Root link name.
            end_link_name (str): End link name.
            collision (bool): If True, get collision mesh origins, else visual mesh origins.

        Returns:
            Dict[str,np.ndarray]: Dictionary of link names and mesh origins.
        """
        import transformations

        link_names = self.chain_link_names(
            root_link_name=root_link_name, end_link_name=end_link_name
        )
        mesh_origins = {}
        for link_name in link_names:
            link: urdf_parser_py.urdf.Link = self._robot.link_map[link_name]
            if collision:
                if link.collision is None:
                    continue
                link_origin = link.collision.origin
            else:
                if link.visual is None:
                    continue
                link_origin = link.visual.origin
            origin = transformations.euler_matrix(
                link_origin.rpy[0], link_origin.rpy[1], link_origin.rpy[2], "sxyz"
            )
            origin[:3, 3] = link_origin.xyz
            mesh_origins[link_name] = origin
        return mesh_origins

    def _verify_links_in_chain(self, root_link_name: str, end_link_name: str) -> None:
        if not self._robot:
            raise RuntimeError("Robot not initialized.")
        link_names = [link.name for link in self._robot.links]
        if not end_link_name in link_names:
            raise ValueError(f"Link {end_link_name} not in robot.")
        if not root_link_name in link_names:
            raise ValueError(f"Link {root_link_name} not in robot.")

    @property
    def urdf(self) -> str:
        r"""Get URDF string."""
        if self._urdf is None:
            raise ValueError("URDF not loaded.")
        return self._urdf

    @property
    def robot(self) -> urdf_parser_py.urdf.Robot:
        r"""Get robot object."""
        return self._robot


def find_files(path: str, pattern: str = "image_*.png") -> List[str]:
    r"""Find files in a directory.

    Args:
        path (str): Path to the directory.
        pattern (str): Pattern to match. Warn: The sorting key strictly assumes that pattern includes '_{number}.ext'.

    Returns:
        List[str]: File names.
    """
    path = pathlib.Path(path)
    image_paths = list(path.glob(pattern))
    return sorted(
        [image_path.name for image_path in image_paths],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )


class MonocularDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        image_pattern: str,
        joint_states_path: str,
        joint_states_pattern: str,
    ):
        self._images_path = images_path
        self._image_files = find_files(images_path, image_pattern)
        self._joint_states_path = joint_states_path
        self._joint_states_files = find_files(joint_states_path, joint_states_pattern)

        rich.print("Found the following files:")
        rich.print(f"Images: {self._image_files}")
        rich.print(f"Joint states: {self._joint_states_files}")

        if len(self._image_files) != len(self._joint_states_files):
            raise ValueError(
                f"Number of images '{len(self._image_files)}' and joint states '{len(self._joint_states_files)}' do not match."
            )

        if len(self._image_files) == 0:
            raise ValueError("No images found.")

        if len(self._joint_states_files) == 0:
            raise ValueError("No joint states found.")

        for image_file, joint_states_file in zip(
            self._image_files, self._joint_states_files
        ):
            if (
                image_file.split("_")[-1].split(".")[0]
                != joint_states_file.split("_")[-1].split(".")[0]
            ):
                raise ValueError(
                    f"Image file index '{image_file}' and joint states file index '{joint_states_file}' do not match."
                )

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        image_file = self._image_files[idx]
        joint_states_file = self._joint_states_files[idx]
        image = cv2.imread(os.path.join(self._images_path, image_file))
        joint_states = np.load(os.path.join(self._joint_states_path, joint_states_file))
        return image, joint_states, image_file


def parse_camera_info(camera_info_file: str) -> Tuple[int, int, np.ndarray]:
    r"""Parse camera info file.

    Args:
        camera_info_file (str): Absolute path to the camera info file.

    Returns:
        Tuple[int,int,np.ndarray]:
            - Height of the image.
            - Width of the image.
            - Intrinsic matrix of shape 3x3.
    """
    with open(camera_info_file, "r") as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    height = camera_info["height"]
    width = camera_info["width"]
    if len(camera_info["k"]) != 9:
        raise ValueError("Camera matrix must be 3x3.")
    intrinsic_matrix = np.array(camera_info["k"]).reshape(3, 3)
    return height, width, intrinsic_matrix


def parse_hydra_data(
    path: str,
    joint_states_files: List[str],
    mask_files: List[str],
    depth_files: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""Parse data for Hydra registration.

    Args:
        path (str): Path to the data.
        joint_states_files (List[str]): Joint states files.
        mask_files (List[str]): Mask files.
        depth_files (List[str]): Depth files. Note that depth values are expected in meters.

    Returns:
        Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray]]:
            - Joint states.
            - Masks of shape HxW.
            - Point clouds of shape HxWx3.
    """
    if len(joint_states_files) == 0 or len(mask_files) == 0 or len(depth_files) == 0:
        raise ValueError("No files found.")
    if len(joint_states_files) != len(mask_files) or len(joint_states_files) != len(
        depth_files
    ):
        raise ValueError(
            f"Number of files do not match. Got {len(joint_states_files)} joint state files, {len(mask_files)} mask files, and {len(depth_files)} depth files."
        )

    rich.print("Parsing the following files:")
    rich.print(f"Joint states: {joint_states_files}")
    rich.print(f"Masks: {mask_files}")
    rich.print(f"Depths: {depth_files}")

    # load data
    joint_states = [
        np.load(os.path.join(path, joint_state_file))
        for joint_state_file in joint_states_files
    ]
    masks = [
        cv2.imread(os.path.join(path, mask_file), cv2.IMREAD_GRAYSCALE)
        for mask_file in mask_files
    ]
    depths = [np.load(os.path.join(path, depth_file)) for depth_file in depth_files]
    if not all([mask.dtype == np.uint8 for mask in masks]):
        raise ValueError("Masks must be of type np.uint8.")
    if not all([np.all(mask >= 0) and np.all(mask <= 255) for mask in masks]):
        raise ValueError("Masks must be in the range [0, 255].")
    if not all(
        [mask.shape[:2] == depth.shape[:2] for mask, depth in zip(masks, depths)]
    ):
        raise ValueError("Mask and depth shapes do not match.")
    if not all(mask.ndim == 2 for mask in masks):
        raise ValueError("Masks must be 2D.")
    if not all(depth.ndim == 2 for depth in depths):
        raise ValueError("Depths must be 2D.")
    return joint_states, masks, depths


def parse_mono_data(
    path: str,
    image_files: List[str],
    joint_states_files: List[str],
    mask_files: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    r"""Parse monocular data.

    Args:
        path (str): Path to the data.
        image_files (List[str]): Image files.
        joint_states_files (List[str]): Joint states files.
        mask_files (List[str]): Mask files.

    Returns:
        Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray]]:
            - Images of shape HxWx3.
            - Joint states.
            - Masks of shape HxW.
    """
    if len(image_files) != len(joint_states_files) or len(image_files) != len(
        mask_files
    ):
        raise ValueError("Number of images, joint states, masks do not match.")

    rich.print("Parsing the following files:")
    rich.print(f"Images: {image_files}")
    rich.print(f"Joint states: {joint_states_files}")
    rich.print(f"Masks: {mask_files}")

    images = [cv2.imread(os.path.join(path, file)) for file in image_files]
    joint_states = [np.load(os.path.join(path, file)) for file in joint_states_files]
    masks = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in mask_files
    ]
    if not all([mask.dtype == np.uint8 for mask in masks]):
        raise ValueError("Masks must be of type np.uint8.")
    if not all([np.all(mask >= 0) and np.all(mask <= 255) for mask in masks]):
        raise ValueError("Masks must be in the range [0, 255].")
    if not all(
        [mask.shape[:2] == image.shape[:2] for mask, image in zip(masks, images)]
    ):
        raise ValueError("Mask and image shapes do not match.")
    if not all(mask.ndim == 2 for mask in masks):
        raise ValueError("Masks must be 2D.")
    if not all(image.ndim == 3 for image in images):
        raise ValueError("Images must be 3D.")
    if not all(image.shape[-1] == 3 for image in images):
        raise ValueError("Images must have 3 channels")
    return images, joint_states, masks


def parse_stereo_data(
    path: str,
    left_image_files: List[str],
    right_image_files: List[str],
    joint_states_files: List[str],
    left_mask_files: List[str],
    right_mask_files: List[str],
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    r"""Parse stereo data.

    Args:
        path (str): Path to the data.
        left_image_files (List[str]): Left image files.
        right_image_files (List[str]): Right image files.
        joint_states_files (List[str]): Joint states files.
        left_mask_files (List[str]): Left mask files.
        right_mask_files (List[str]): Right mask files.

    Returns:
        Tuple[List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray],List[np.ndarray]]:
            - Left images of shape HxWx3.
            - Right images of shape HxWx3.
            - Joint states.
            - Left masks of shape HxW.
            - Right masks of shape HxW.
    """
    if (
        len(left_image_files) != len(right_image_files)
        or len(left_image_files) != len(joint_states_files)
        or len(left_image_files) != len(left_mask_files)
        or len(left_image_files) != len(right_mask_files)
    ):
        raise ValueError(
            "Number of left / right images, joint states, left / right masks do not match."
        )

    rich.print("Parsing the following files:")
    rich.print(f"Left images: {left_image_files}")
    rich.print(f"Right images: {right_image_files}")
    rich.print(f"Joint states: {joint_states_files}")
    rich.print(f"Left masks: {left_mask_files}")
    rich.print(f"Right masks: {right_mask_files}")

    left_images = [cv2.imread(os.path.join(path, file)) for file in left_image_files]
    right_images = [cv2.imread(os.path.join(path, file)) for file in right_image_files]
    joint_states = [np.load(os.path.join(path, file)) for file in joint_states_files]
    left_masks = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in left_mask_files
    ]
    right_masks = [
        cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        for file in right_mask_files
    ]
    if not all([mask.dtype == np.uint8 for mask in left_masks]):
        raise ValueError("Left masks must be of type np.uint8.")
    if not all([np.all(mask >= 0) and np.all(mask <= 255) for mask in left_masks]):
        raise ValueError("Left masks must be in the range [0, 255].")
    if not all([mask.dtype == np.uint8 for mask in right_masks]):
        raise ValueError("Left masks must be of type np.uint8.")
    if not all([np.all(mask >= 0) and np.all(mask <= 255) for mask in right_masks]):
        raise ValueError("Left masks must be in the range [0, 255].")
    if not all(mask.ndim == 2 for mask in left_masks):
        raise ValueError("Left masks must be 2D.")
    if not all(image.ndim == 3 for image in left_images):
        raise ValueError("Left images must be 3D.")
    if not all(image.shape[-1] == 3 for image in left_images):
        raise ValueError("Left images must have 3 channels")
    if not all(mask.ndim == 2 for mask in right_masks):
        raise ValueError("Right masks must be 2D.")
    if not all(image.ndim == 3 for image in right_images):
        raise ValueError("Right images must be 3D.")
    if not all(image.shape[-1] == 3 for image in right_images):
        raise ValueError("Right images must have 3 channels")
    return left_images, right_images, joint_states, left_masks, right_masks

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List

import kinpy
import numpy as np
import pyvista
import transformations as tf
import trimesh
import xacro
from ament_index_python import get_package_share_directory
from kinpy.chain import Chain
from pyvista import pyvista_ndarray


class MeshifyRobot:
    chain: Chain
    meshes: List[pyvista.DataSet]

    def __init__(self, urdf: str, resolution: str = "collision") -> None:
        self.chain = self._load_chain(urdf)
        self.joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(self.joint_names)
        self.paths, self.link_names = self._get_mesh_paths(urdf, resolution)
        self.meshes = self._load_meshes(self.paths)

    def transformed_meshes(self, q: np.ndarray) -> List[pyvista.DataSet]:
        zero_tf_dict = self._get_transforms(np.zeros(self.dof))
        tf_dict = self._get_transforms(q)

        meshes = deepcopy(self.meshes)
        for mesh, link in zip(meshes, self.link_names):
            # zero transform
            r0 = tf.quaternion_matrix(zero_tf_dict[link].rot)
            t0 = tf.translation_matrix(zero_tf_dict[link].pos)
            ht0 = np.eye(4)
            ht0[:3, :3] = r0[:3, :3]
            ht0[:3, 3] = t0[:3, 3]

            # desired transform
            r = tf.quaternion_matrix(tf_dict[link].rot)
            t = tf.translation_matrix(tf_dict[link].pos)
            ht = np.eye(4)
            ht[:3, :3] = r[:3, :3]
            ht[:3, 3] = t[:3, 3]

            mesh.transform(ht @ np.linalg.inv(ht0))
        return meshes

    def plot_meshes(
        self, meshes, background_color: str = "gray", mesh_color: str = "white"
    ) -> None:
        plotter = pyvista.Plotter()
        plotter.background_color = background_color
        for mesh in meshes:
            plotter.add_mesh(mesh, color=mesh_color)
        plotter.show()

    def plot_point_clouds(
        self, meshes, background_color: str = "gray", point_color: str = "white"
    ) -> None:
        point_clouds = self.meshes_to_point_clouds(meshes)
        point_clouds = [pyvista.PolyData(point_cloud) for point_cloud in point_clouds]
        plotter = pyvista.Plotter()
        plotter.background_color = background_color
        for point_cloud in point_clouds:
            plotter.add_mesh(point_cloud, point_size=1, color=point_color)
        plotter.show()

    def meshes_to_point_clouds(
        self, meshes: List[pyvista.DataSet]
    ) -> List[pyvista_ndarray]:
        point_clouds = [self.mesh_to_point_cloud(mesh) for mesh in meshes]
        return point_clouds

    def meshes_to_point_cloud(self, meshes: List[pyvista.DataSet]) -> np.ndarray:
        point_clouds = self.meshes_to_point_clouds(meshes)
        point_cloud = np.concatenate(point_clouds, axis=0)
        return point_cloud

    def mesh_to_point_cloud(self, mesh: pyvista.DataSet) -> pyvista_ndarray:
        point_cloud = mesh.points
        return point_cloud

    def remove_inner_points(self, point_cloud: np.ndarray, alpha: float):
        raise DeprecationWarning("To be re-added.")

    def _sub_sample(self, data: np.ndarray, N: int):
        indices = np.random.choice(data.shape[0], N, replace=False)
        sampled_points = data[indices]
        return sampled_points

    def _get_transforms(self, q: np.ndarray) -> Dict[str, kinpy.Transform]:
        transforms = self.chain.forward_kinematics(q)
        return transforms

    def _get_mesh_paths(self, urdf: str, resolution: str = "collision"):
        paths = []
        names = []

        def handle_package_path(package: str, filename: str):
            package_path = get_package_share_directory(package)
            return os.path.join(package_path, filename)

        robot = ET.fromstring(urdf)
        for link in robot.findall("link"):
            visual = link.find(resolution)
            if visual:
                name = link.attrib["name"]
                geometry = visual.find("geometry")
                mesh = geometry.find("mesh")
                filename = mesh.attrib["filename"]

                if filename.startswith("package://"):
                    filename = filename.replace("package://", "")
                    package, filename = filename.split("/", 1)
                    path = handle_package_path(package, filename)
                    names.append(name)
                    paths.append(path)
        return paths, names

    def _load_mesh(self, path: str):
        print(f"Loading mesh from {path}")
        if path.endswith(".stl"):
            mesh = pyvista.read(path)
        elif path.endswith(".dae"):
            scene = trimesh.load_mesh(path)
            vertices = []
            faces = []
            for geometry in scene.geometry.values():
                vertices.append(geometry.vertices)
                faces.append(geometry.faces)
            vertices = np.concatenate(vertices, axis=0).tolist()
            faces = np.concatenate(faces, axis=0).tolist()
            mesh = pyvista.PolyData(vertices, faces)
        else:
            raise NotImplementedError(f"File type {path} not supported.")
        return mesh

    def _load_meshes(self, paths: List[str]):
        meshes = [self._load_mesh(path) for path in paths]
        return meshes

    def _load_chain(self, urdf: str) -> Chain:
        chain = kinpy.build_chain_from_urdf(urdf)
        return chain


if __name__ == "__main__":
    urdf = xacro.process(
        os.path.join(
            get_package_share_directory("lbr_description"), "urdf/med7/med7.urdf.xacro"
        )
    )

    meshify_robot = MeshifyRobot(urdf)

    for i in range(10):
        q = np.random.uniform(-np.pi / 2, np.pi / 2, meshify_robot.dof)
        meshes = meshify_robot.transformed_meshes(q)
        meshify_robot.plot_meshes(meshes)
        meshify_robot.plot_point_clouds(meshes)

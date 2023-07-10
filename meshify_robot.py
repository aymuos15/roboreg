import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List

import alphashape
from scipy.spatial import Delaunay
import kinpy
import numpy as np
import pyvista
import transformations as tf
import xacro
from ament_index_python import get_package_share_directory
from kinpy.chain import Chain
from pyvista import pyvista_ndarray
from shapely.geometry import Point
from collections import defaultdict
from shapely.ops import cascaded_union


class MeshifyRobot:
    chain: Chain
    meshes: List[pyvista.DataSet]

    def __init__(self, urdf: str) -> None:
        self.chain = self._load_chain(urdf)
        self.joint_names = self.chain.get_joint_parameter_names(exclude_fixed=True)
        self.dof = len(self.joint_names)
        self.paths, self.link_names = self._get_mesh_paths(urdf)
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

    def plot_meshes(self, meshes) -> None:
        plottter = pyvista.Plotter()
        for mesh in meshes:
            plottter.add_mesh(mesh)
        plottter.show()

    def plot_point_clouds(self, meshes) -> None:
        point_clouds = self.meshes_to_point_clouds(meshes)
        point_clouds = [pyvista.PolyData(point_cloud) for point_cloud in point_clouds]
        plottter = pyvista.Plotter()
        for point_cloud in point_clouds:
            plottter.add_mesh(point_cloud, point_size=1)
        plottter.show()

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

    def alpha_shape_from_point_cloud(self, point_cloud: np.ndarray, alpha: float):
        """
        Taken from https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d

        Compute the alpha shape (concave hull) of a set of 3D points.
        Parameters:
            point_cloud - np.array of shape (n,3) points.
            alpha - alpha value.
        return
            outer surface vertex indices, edge indices, and triangle indices
        """

        tetra = Delaunay(point_cloud)
        # Find radius of the circumsphere.
        # By definition, radius of the sphere fitting inside the tetrahedral needs
        # to be smaller than alpha value
        # http://mathworld.wolfram.com/Circumsphere.html
        tetrapos = np.take(point_cloud, tetra.vertices, axis=0)
        normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
        ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
        a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
        Dx = np.linalg.det(
            np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2)
        )
        Dy = -np.linalg.det(
            np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2)
        )
        Dz = np.linalg.det(
            np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2)
        )
        c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
        r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * np.abs(a))

        # Find tetrahedrals
        tetras = tetra.vertices[r < alpha, :]
        # triangles
        tri_comb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        triangles = tetras[:, tri_comb].reshape(-1, 3)
        triangles = np.sort(triangles, axis=1)
        # Remove triangles that occurs twice, because they are within shapes
        triangle_dict = defaultdict(int)
        for tri in triangles:
            triangle_dict[tuple(tri)] += 1
        triangles = np.array([tri for tri in triangle_dict if triangle_dict[tri] == 1])
        # edges
        edge_comb = np.array([(0, 1), (0, 2), (1, 2)])
        edges = triangles[:, edge_comb].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        vertices = np.unique(edges)
        return vertices, edges, triangles

    def remove_inner_points(self, point_cloud: np.ndarray, alpha: float):
        vertices, edges, triangles = self.alpha_shape_from_point_cloud(
            point_cloud, alpha
        )
        return point_cloud[vertices]

    def _sub_sample(self, data: np.ndarray, N: int):
        indices = np.random.choice(data.shape[0], N, replace=False)
        sampled_points = data[indices]
        return sampled_points

    def _get_transforms(self, q: np.ndarray) -> Dict[str, kinpy.Transform]:
        transforms = self.chain.forward_kinematics(q)
        return transforms

    def _get_mesh_paths(self, urdf: str):
        paths = []
        names = []

        def handle_package_path(package: str, filename: str):
            package_path = get_package_share_directory(package)
            return os.path.join(package_path, filename)

        robot = ET.fromstring(urdf)
        for link in robot.findall("link"):
            visual = link.find("visual")
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
        mesh = pyvista.read(path)
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
        q = np.random.uniform(-np.pi, np.pi, meshify_robot.dof)
        meshes = meshify_robot.transformed_meshes(q)
        meshify_robot.plot_meshes(meshes)
        meshify_robot.plot_point_clouds(meshes)

        # convex hull
        point_cloud = meshify_robot.meshes_to_point_cloud(meshes)
        vertices, edges, triangles = meshify_robot.alpha_shape_from_point_cloud(
            point_cloud, alpha=0.2
        )
        point_cloud = point_cloud[vertices]

        # plot
        point_cloud = pyvista.PolyData(point_cloud)
        plotter = pyvista.Plotter()
        plotter.add_mesh(point_cloud, point_size=1)
        plotter.show()

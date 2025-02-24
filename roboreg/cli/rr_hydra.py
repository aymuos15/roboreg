import argparse
import os

import numpy as np
import rich
import torch

from roboreg.differentiable import Robot
from roboreg.hydra_icp import hydra_centroid_alignment, hydra_robust_icp
from roboreg.io import URDFParser, find_files, parse_camera_info, parse_hydra_data
from roboreg.util import (
    RegistrationVisualizer,
    clean_xyz,
    compute_vertex_normals,
    depth_to_xyz,
    from_homogeneous,
    generate_ht_optical,
    mask_extract_extended_boundary,
    to_homogeneous,
)


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--camera-info-file",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the data.")
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="image_*_mask.png",
        help="Mask file pattern.",
    )
    parser.add_argument(
        "--depth-pattern",
        type=str,
        default="depth_*.npy",
        help="Depth file pattern. Note that depth values are expected in meters.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--ros-package",
        type=str,
        default="lbr_description",
        help="Package where the URDF is located.",
    )
    parser.add_argument(
        "--xacro-path",
        type=str,
        default="urdf/med7/med7.xacro",
        help="Path to the xacro file, relative to --ros-package.",
    )
    parser.add_argument(
        "--root-link-name",
        type=str,
        default="",
        help="Root link name. If unspecified, the first link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--end-link-name",
        type=str,
        default="",
        help="End link name. If unspecified, the last link with mesh will be used, which may cause errors.",
    )
    parser.add_argument(
        "--visual-meshes",
        action="store_true",
        help="If set, visual meshes will be used instead of collision meshes.",
    )
    parser.add_argument(
        "--depth-conversion-factor",
        type=float,
        default=1.0,
        help="Conversion factor for depth. Computes z = depth / conversion_factor e.g. to covert from millimeter to meter.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.01,
        help="Minimum depth value.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=2.0,
        help="Maximum depth value.",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=5000,
        help="Number of points to sample from robot mesh.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.1,
        help="Maximum distance between two points to be considered as a correspondence.",
    )
    parser.add_argument(
        "--outer-max-iter",
        type=int,
        default=50,
        help="Maximum number of outer iterations.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=10,
        help="Maximum number of inner iterations.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="HT_hydra_robust.npy",
        help="Output file name. Relative to the path.",
    )
    parser.add_argument(
        "--no-boundary",
        action="store_true",
        help="Do not apply dilation / erosion to the mask.",
    )
    parser.add_argument(
        "--dilation-kernel-size",
        type=int,
        default=3,
        help="Dilation kernel size for mask boundary. Larger value will result in larger boundary.",
    )
    parser.add_argument(
        "--erosion-kernel-size",
        type=int,
        default=10,
        help="Erosion kernel size for mask boundary. Larger value will result in larger boundary. The closer the robot, the larger the recommended kernel size.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    joint_states_files = find_files(args.path, args.joint_states_pattern)
    mask_files = find_files(args.path, args.mask_pattern)
    depth_files = find_files(args.path, args.depth_pattern)
    joint_states, masks, depths = parse_hydra_data(
        path=args.path,
        joint_states_files=joint_states_files,
        mask_files=mask_files,
        depth_files=depth_files,
    )
    height, width, intrinsics = parse_camera_info(args.camera_info_file)

    # instantiate kinematics
    urdf_parser = URDFParser()
    urdf_parser.from_ros_xacro(ros_package=args.ros_package, xacro_path=args.xacro_path)
    root_link_name = args.root_link_name
    end_link_name = args.end_link_name
    if root_link_name == "":
        root_link_name = urdf_parser.link_names_with_meshes(visual=args.visual_meshes)[
            0
        ]
        rich.print(
            f"Root link name not provided. Using the first link with mesh: '{root_link_name}'."
        )
    if end_link_name == "":
        end_link_name = urdf_parser.link_names_with_meshes(visual=args.visual_meshes)[
            -1
        ]
        rich.print(
            f"End link name not provided. Using the last link with mesh: '{end_link_name}'."
        )

    # instantiate robot
    batch_size = len(joint_states)
    robot = Robot(
        urdf_parser=urdf_parser,
        root_link_name=root_link_name,
        end_link_name=end_link_name,
        visual=args.visual_meshes,
        batch_size=batch_size,
    )

    # perform forward kinematics
    joint_states = torch.tensor(
        np.array(joint_states), dtype=torch.float32, device=device
    )
    robot.configure(joint_states)

    # turn depths into xyzs
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    depths = torch.tensor(np.array(depths), dtype=torch.float32, device=device)
    xyzs = depth_to_xyz(
        depth=depths,
        intrinsics=intrinsics,
        z_min=args.z_min,
        z_max=args.z_max,
        conversion_factor=args.depth_conversion_factor,
    )

    # flatten BxHxWx3 -> Bx(H*W)x3
    xyzs = xyzs.view(-1, height * width, 3)
    xyzs = to_homogeneous(xyzs)
    ht_optical = generate_ht_optical(xyzs.shape[0], dtype=torch.float32, device=device)
    xyzs = torch.matmul(xyzs, ht_optical.transpose(-1, -2))
    xyzs = from_homogeneous(xyzs)

    # unflatten
    xyzs = xyzs.view(-1, height, width, 3)
    xyzs = [xyz.squeeze() for xyz in xyzs.cpu().numpy()]

    # mesh vertices to list
    mesh_vertices = from_homogeneous(robot.configured_vertices)
    mesh_vertices = [mesh_vertices[i].contiguous() for i in range(batch_size)]
    mesh_normals = []
    for i in range(batch_size):
        mesh_normals.append(
            compute_vertex_normals(vertices=mesh_vertices[i], faces=robot.faces)
        )

    # clean observed vertices and turn into tensor
    observed_vertices = [
        torch.tensor(
            clean_xyz(
                xyz=xyz,
                mask=(
                    mask
                    if args.no_boundary
                    else mask_extract_extended_boundary(
                        mask,
                        dilation_kernel=np.ones(
                            [args.dilation_kernel_size, args.dilation_kernel_size]
                        ),
                        erosion_kernel=np.ones(
                            [args.erosion_kernel_size, args.erosion_kernel_size]
                        ),
                    )
                ),
            ),
            dtype=torch.float32,
            device=device,
        )
        for xyz, mask in zip(xyzs, masks)
    ]

    # sample N points per mesh
    for i in range(batch_size):
        idx = torch.randperm(mesh_vertices[i].shape[0])[: args.number_of_points]
        mesh_vertices[i] = mesh_vertices[i][idx]
        mesh_normals[i] = mesh_normals[i][idx]

    HT_init = hydra_centroid_alignment(observed_vertices, mesh_vertices)
    HT = hydra_robust_icp(
        HT_init,
        observed_vertices,
        mesh_vertices,
        mesh_normals,
        max_distance=args.max_distance,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
    )

    # visualize
    visualizer = RegistrationVisualizer()
    visualizer(mesh_vertices=mesh_vertices, observed_vertices=observed_vertices)
    visualizer(
        mesh_vertices=mesh_vertices,
        observed_vertices=observed_vertices,
        HT=torch.linalg.inv(HT),
    )

    # to numpy
    HT = HT.cpu().numpy()
    np.save(os.path.join(args.path, args.output_file), HT)


if __name__ == "__main__":
    main()

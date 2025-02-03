import argparse
import os
import pathlib

import cv2
import numpy as np
import torch
from rich import progress
from torch.utils.data import DataLoader

from roboreg.io import MonocularDataset
from roboreg.util import overlay_mask
from roboreg.util.factories import create_robot_scene, create_virtual_camera


def args_factory() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for rendering. For batch_size > 1, the last batch may be dropped.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of workers for data loading."
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
        "--camera-info-file",
        type=str,
        required=True,
        help="Path to the camera parameters, <path_to>/camera_info.yaml.",
    )
    parser.add_argument(
        "--extrinsics-file",
        type=str,
        required=True,
        help="Homogeneous transform from base to camera frame, <path_to>/HT_hydra_robust.npy.",
    )
    parser.add_argument(
        "--images-path", type=str, required=True, help="Path to the images."
    )
    parser.add_argument(
        "--joint-states-path", type=str, required=True, help="Path to the joint states."
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="image_*.png",
        help="Image file pattern.",
    )
    parser.add_argument(
        "--joint-states-pattern",
        type=str,
        default="joint_states_*.npy",
        help="Joint state file pattern.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path.",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["r", "g", "b"],
        default="b",
        help="Color channel to overlay the render.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Number of concurrent compilation jobs for nvdiffrast. Only relevant on first run.",
    )
    return parser.parse_args()


def main():
    args = args_factory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["MAX_JOBS"] = str(args.max_jobs)  # limit number of concurrent jobs
    camera = {
        "camera": create_virtual_camera(
            camera_info_file=args.camera_info_file,
            extrinsics_file=args.extrinsics_file,
            device=device,
        )
    }
    scene = create_robot_scene(
        batch_size=args.batch_size,
        ros_package=args.ros_package,
        xacro_path=args.xacro_path,
        root_link_name=args.root_link_name,
        end_link_name=args.end_link_name,
        cameras=camera,
        device=device,
        visual=args.visual_meshes,
    )
    dataset = MonocularDataset(
        images_path=args.images_path,
        image_pattern=args.image_pattern,
        joint_states_path=args.joint_states_path,
        joint_states_pattern=args.joint_states_pattern,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    output_path = pathlib.Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    for images, joint_states, image_files in progress.track(
        dataloader, description="Rendering..."
    ):
        # pre-process
        joint_states = joint_states.to(dtype=torch.float32, device=device)

        # configure robot
        scene.robot.configure(joint_states)

        # render
        renders = scene.observe_from(list(scene.cameras.keys())[0])

        # save
        images = images.numpy()
        renders = (renders * 255.0).squeeze(-1).cpu().numpy().astype(np.uint8)
        for render, image, image_file in zip(renders, images, image_files):
            image_stem = pathlib.Path(image_file).stem
            image_suffix = pathlib.Path(image_file).suffix
            cv2.imwrite(
                os.path.join(
                    str(output_path.absolute()),
                    f"overlay_render_{image_stem + image_suffix}",
                ),
                overlay_mask(image, render, args.color, scale=1.0),
            )


if __name__ == "__main__":
    main()

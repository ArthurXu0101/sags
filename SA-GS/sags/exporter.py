# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox
from sags.model.sags import Sags
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path #= Path("/home/xiongbutian/workspace/sags/outputs/hav6w-ours-small/sags/2024-11-06_222559/config.yml")
    """Path to the config YAML file."""
    output_dir: Path #= Path("exports/splat")
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """

        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, Sags)

        model: Sags = pipeline.model

        filename = self.output_dir / "splat.ply"

        map_to_tensors = OrderedDict()

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
            map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if self.ply_color_mode == "rgb":
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                colors = (colors * 255).astype(np.uint8)
                map_to_tensors["red"] = colors[:, 0]
                map_to_tensors["green"] = colors[:, 1]
                map_to_tensors["blue"] = colors[:, 2]
            elif self.ply_color_mode == "sh_coeffs":
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            if model.config.sh_degree > 0:
                if self.ply_color_mode == "rgb":
                    CONSOLE.print(
                        "Warning: model has higher level of spherical harmonics, ignoring them and only export rgb."
                    )
                elif self.ply_color_mode == "sh_coeffs":
                    # transpose(1, 2) was needed to match the sh order in Inria version
                    shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                    shs_rest = shs_rest.reshape((n, -1))
                    for i in range(shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

            map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

            scales = model.scales.data.cpu().numpy()
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa

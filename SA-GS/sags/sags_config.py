"""
Street Gaussians configuration file.
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from sags.data.SagsDataParser import SagsDataParserConfig
from sags.data.SagsDataset import SagsDataset
from sags.data.SagsDataManager import SagsDataManager, SagsDataManagerConfig
from sags.model.sags import SagsConfig

sags = MethodSpecification(
    config=TrainerConfig(
        method_name="sags",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 1},
        pipeline=VanillaPipelineConfig(
            datamanager=SagsDataManagerConfig(
                _target=SagsDataManager[SagsDataset],
                dataparser=SagsDataParserConfig(
                    load_pretrained_gs=False,
                    max_2D_matches_per_3D_point=0,
                    orientation_method="up",
                    center_method="poses",
                    auto_scale_poses=True,
                    load_3D_points=True,
                    
                ),
            ),
            model=SagsConfig(
                stop_split_at=10000
                ),
        ),
        optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+wandb",
    ),
    description="Semantic Feature Extraction Script, let us start to do the initialization of models",
)

sags_big = MethodSpecification(
    config=TrainerConfig(
        method_name="sags-big",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 1},
        pipeline=VanillaPipelineConfig(
            datamanager=SagsDataManagerConfig(
                _target=SagsDataManager[SagsDataset],
                dataparser=SagsDataParserConfig(
                    load_pretrained_gs=False,
                    max_2D_matches_per_3D_point=0,
                ),
            ),
            model=SagsConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
                densify_grad_thresh=0.0006,
                ),
        ),
        optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+wandb",
    ),
    description="Large version of sags, let us start to do the initialization of models",
)

sags_refine = MethodSpecification(
    config=TrainerConfig(
        method_name="sags-refine",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=20000, 
        max_num_iterations=20000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 1},
        pipeline=VanillaPipelineConfig(
            datamanager=SagsDataManagerConfig(
                _target=SagsDataManager[SagsDataset],
                dataparser=SagsDataParserConfig(
                    load_pretrained_gs=True,
                    max_2D_matches_per_3D_point=0,
                ),
            ),
            model=SagsConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
                densify_grad_thresh=0.0006,
                stop_split_at=0
                ),
        ),
        optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer+wandb",
    ),
    description="load pretrained GS for refinement, default sags-config",
)
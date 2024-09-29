"""
Street Gaussians configuration file.
"""
from pathlib import Path

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


from semanticField.data.SemanticFieldDataParser import SemanticDataParserConfig
from semanticField.data.SemanticFieldDataManager import SemanticDataManagerConfig, SemanticDatamanager
from semanticField.data.SemanticFieldDataset import SemanticFieldDataset
from semanticField.model.semanticField import SemanticFieldConfig

semanticField = MethodSpecification(
    config=TrainerConfig(
        method_name="semantic-field",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 1},
        pipeline=VanillaPipelineConfig(
            datamanager=SemanticDataManagerConfig(
                _target=SemanticDatamanager[SemanticFieldDataset],
                dataparser=SemanticDataParserConfig(
                    load_pretrained_gs=True,
                    max_2D_matches_per_3D_point=0,
                ),
            ),
            model=SemanticFieldConfig(),
        ),
        optimizers={
            "features": {
                "optimizer": AdamOptimizerConfig(lr=0.0035, eps=1e-15),
                "scheduler": None,
            },
            "colors": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_legacy+tensorboard",
    ),
    description="Semantic Feature Extraction Script, let us start to do the initialization of models",
)
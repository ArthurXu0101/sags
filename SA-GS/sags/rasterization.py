import torch
from feature_splat.rendering import fully_fused_projection, isect_tiles, isect_offset_encode, pxiel_gaussian_weight_retrival
import math
def rasterization(
    means: torch.Tensor,
    quats:torch.Tensor,
    scales:torch.Tensor,
    opacities:torch.Tensor,
    viewmats:torch.Tensor,  # [1, 4, 4]
    Ks:torch.Tensor,  # [1, 3, 3]
    width:int,
    height:int,
    tile_size:int,
    packed=True,
    near_plane=0.01,
    far_plane=1e10,
    render_mode='RGB',
    rasterize_mode='antialiased',
):
    """
        Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians. It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, width, height, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, width, height, 1].

        **meta**: A dictionary of intermediate results of the rasterization.
        """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=0.3,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
    )


    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None
    
    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )

    isect_offset = isect_offset_encode(
        isect_ids=isect_ids,
        n_cameras=C,
        tile_width=tile_width,
        tile_height=tile_height
    )

    flatten_ids_gaussian_correspondance, pixelIds, weight = pxiel_gaussian_weight_retrival(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        isect_offsets=isect_offset,
        flatten_ids=flatten_ids,
        packed=True
    )

    pixel_location = torch.stack([pixelIds//width, pixelIds%width], dim=1)
    
    return gaussian_ids[flatten_ids_gaussian_correspondance], pixel_location, weight

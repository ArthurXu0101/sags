from sags.data.SagsDataParser import SagsDataParserConfig, SagsDataParser
from gsplat.rendering import rasterization
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

transform_matrix = torch.tensor([
        [ 1.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  1.]
], dtype=torch.float32).cuda()


BLOCK_WIDTH = 16
# cameras can be iterate
if __name__ == "__main__":
    config = SagsDataParserConfig(
        data=Path('/data2/butian/GauUscene/HAV_COLMAP/colmap/0')
    )
    
    parser = SagsDataParser(config=config)
    
    output = parser.get_dataparser_outputs()
    
    camera = output.cameras[0:1]
    
    c2w = torch.eye(4).cuda()
    
    c2w[:3] = camera.camera_to_worlds.cuda()
    #c2w = transform_matrix@c2w
    viewmat = get_viewmat(c2w.unsqueeze(0))
    viewmat = viewmat
    
    
    point_cloud: dict = output.metadata['points3D_xyz']
    
    means = point_cloud['means'].cuda()
    scales = point_cloud['scales'].T.cuda()
    quats = point_cloud['quats'].T.cuda()
    opacities = point_cloud['opacities'].T.cuda()
    feature_dc = point_cloud['feature_dc'].T.cuda()
    feature_rest = point_cloud['feature_rest'].T.reshape(feature_dc.shape[0],-1,3).cuda()
    colors_crop = torch.cat((feature_dc[:, None, :], feature_rest), dim=1)
    
    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())
    print(output.image_filenames[0])
    
    render, alpha, info = rasterization(
        means=means,
        quats=quats / quats.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        colors=colors_crop,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K,  # [1, 3, 3]
        width=W,
        height=H,
        tile_size=BLOCK_WIDTH,
        packed=True,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sh_degree=3,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode='classic',
        # set some threshold to disregrad small gaussians for faster rendering.
        # radius_clip=3.0,
    )
    render.clamp(0, 1)
    render:torch.Tensor = render.squeeze()
    
    image = TF.to_pil_image(render.permute(2,0,1))  # Convert to PIL image
    image.save("output_image_1.jpg")    
    
    
    
    
    
    

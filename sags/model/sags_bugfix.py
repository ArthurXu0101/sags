from __future__ import annotations
import torch

import sys
sys.path.append('..')
from model import sags
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aabb_instance = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32).to(device)
    scenebox_instance = SceneBox(aabb_instance)
    num_train_data_instance = 1
    size = 500
    sagscfg_instance = sags.SagsConfig()

    try:
        # Initialize the model and move it to the correct device
        sags_model = sags.Sags(sagscfg_instance, scenebox_instance, num_train_data_instance).to(device)
        sags_model.populate_modules()
        sags_model.to(device)  # Ensure all modules are on the correct device
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Model initialization error:\n{e}")

    #print(sags_model.gauss_params)

    try:
        # Define camera parameters
        batch_size = 1
        # Create a batch_size x 4 x 4 tensor as the homogeneous transformation matrix
        camera_to_world = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 4, 4]
        # Set the translation part for the camera to [1.0, 2.0, 3.0]
        camera_to_world[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0], device=device)
        fx, fy = torch.tensor(500.0, device=device), torch.tensor(500.0, device=device)  # Focal lengths
        cx, cy = torch.tensor(250.0, device=device), torch.tensor(250.0, device=device)  # Principal point
        width, height = torch.tensor(500, device=device), torch.tensor(500, device=device)  # Image resolution
        camera_type = 0  # Assuming perspective camera

        # Create the camera instance and move it to the correct device
        camera_instance = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            camera_type=camera_type
        ).to(device)
        print("Camera instance created successfully.")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        camera_instance = None

    # Ground truth batch with random data for testing
    gt_imgs_batch = {}
    # Shape (H, W, 3), where 1 image (H, W) is randomly generated with 3 color channels
    gt_imgs_batch['image'] = (torch.rand(size, size, 3) * 255).byte().to(device)
    # Shape (H, W, 1), where 1 mask (H, W) is generated with int semantic label 0-255
    gt_imgs_batch['mask'] = (torch.rand(size, size, 1) * 255).byte().to(device)

    if camera_instance is not None:
        with torch.no_grad():
            try:
                # Generate outputs using the camera
                #outputs_batch = sags_model(gt_imgs_batch)
                outputs_batch = sags_model.get_outputs(camera_instance)
                print("Outputs generated successfully.")
                #print(outputs_batch.keys())  # Print the keys of the generated outputs
            except Exception as e:
                print(f"Error while generating outputs: {e}")
                outputs_batch = None

    #print(sags_model.gaussian_ids)
    #print(sags_model.flatten_ids)
    #print(sags_model.xys)
 
    # Calculate the loss if outputs were generated successfully
    if outputs_batch is not None:
        try:
            loss_dict = sags_model.get_loss_dict(outputs_batch, gt_imgs_batch)
            print(f"Loss calculated: {loss_dict.items()}")
        except Exception as e:
            print(f"Error during loss calculation: {e}")

    print('--test end--')



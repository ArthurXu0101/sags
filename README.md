# SA-GS: Semantic-Aware Gaussian Splatting for Large Scene Reconstruction with Geometry Constrain
This is the unofficial implementation for [SAGS](https://saliteta.github.io/SA-GS-public/).

# Installation
LERF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. 
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/ArthurXu0101/sags.git`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with sags

# Using SAGS
Now that SAGS is installed you can play with it! 

# Preparing the data folder


```
.
├────<Path/to/your/dataset>
│    ├── colmap  
│    │   └── 0 # blocks, can be adding to various blocks
│    ├── geometric_complexity.csv  # generated geometric_complexity
│    ├── images   
│    └── npz  # semantic_masks
```


- Launch training with `ns-train sags --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 
- Connect to the viewer by forwarding the viewer port (we use VSCode to do this), and click the link to `viewer.nerf.studio` provided in the output of the train script

## `sags-big` and `sags-lite` #ToDo
The default settings provided maintain a balance between speed ('sags'), quality, and splat file size, but if you care more about quality than training speed or size, you can decrease the alpha cull threshold (threshold to delete translucent gaussians) and disable culling after 15k steps like so: 'ns-train splatfacto --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False --data <data>'
`sags-big` provides a larger model that uses More Gaussians, Higher Quality.

# defloater
A common artifact in splatting is long, spikey gaussians. PhysGaussian proposes a scale regularizer that encourages gaussians to be more evenly shaped. To enable this, set the 'pipeline.model.use_scale_regularization' flag to True.

### Exporting splats as '.PLY'
Gaussian splats can be exported as a .ply file which are ingestable by a variety of online web viewers. You can do this via the viewer, or
Run 'python SA-GS/sags/exporter.py gaussian-splat --load-config <config> --output-dir exports/splat' 


### COLMAP input camera porcessing
We find input COLMAP camera will be convert to OPEN-GL format, and will be rescale and rotate. Currently, we might need to use OPEN-GL format one, but no need to rescale, centered, and rotate. Let test the rendered result.
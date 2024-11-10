import numpy as np
from plyfile import PlyData
import torch


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    out = {}
    out["means"] = torch.from_numpy(np.vstack([vertices['x'], vertices['y'], vertices['z']]).T)
    out["scales"] = torch.from_numpy(np.vstack([vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]))
    out["quats"] = torch.from_numpy(np.vstack([vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']]))
    out["opacities"] = torch.from_numpy(np.vstack([vertices['opacity']]))
    out["feature_dc"] = torch.from_numpy(np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]))
    out["feature_rest"] = torch.from_numpy(np.vstack([
        vertices['f_rest_0'], vertices['f_rest_1'], vertices['f_rest_2'],vertices['f_rest_3'], 
        vertices['f_rest_4'], vertices['f_rest_5'],vertices['f_rest_6'], vertices['f_rest_7'], 
        vertices['f_rest_8'],vertices['f_rest_9'], vertices['f_rest_10'], vertices['f_rest_11'],
        vertices['f_rest_12'], vertices['f_rest_13'], vertices['f_rest_14'],vertices['f_rest_15'], 
        vertices['f_rest_16'], vertices['f_rest_17'],vertices['f_rest_18'], vertices['f_rest_19'], 
        vertices['f_rest_20'],vertices['f_rest_21'], vertices['f_rest_22'], vertices['f_rest_23'],
        vertices['f_rest_24'], vertices['f_rest_25'], vertices['f_rest_26'],vertices['f_rest_27'], 
        vertices['f_rest_28'], vertices['f_rest_29'],vertices['f_rest_30'], vertices['f_rest_31'], 
        vertices['f_rest_32'],vertices['f_rest_33'], vertices['f_rest_34'], vertices['f_rest_35'],
        vertices['f_rest_36'], vertices['f_rest_37'], vertices['f_rest_38'],vertices['f_rest_39'], 
        vertices['f_rest_40'], vertices['f_rest_41'],vertices['f_rest_42'], vertices['f_rest_43'], 
        vertices['f_rest_44']]
        ))

    return out


a = fetchPly("/data2/butian/GauUscene/CUHK_LOWER_CAMPUS_COLMAP/colmap/0/point_cloud/iteration_30000/point_cloud.ply")
print(a["means"])
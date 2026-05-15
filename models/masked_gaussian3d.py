import torch
import numpy as np
from .gaussian3d import Gaussian3D  # Assumindo que está no mesmo pacote
from utils.graphics import BasicPointCloud

class MaskedGaussian3D(Gaussian3D):

    def __init__(self, cfg, log, work_dir, debug=False):
        super().__init__(cfg, log, work_dir, debug)
        self._lc = torch.empty(0)

    def initialize(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        super().initialize(pcd, spatial_lr_scale)
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self._lc = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['mask_params'] = (self._lc,)  # type: ignore
        return state_dict

    def load_state_dict(self, state_dict, optim_cfg):
        super().load_state_dict(state_dict, optim_cfg)
        self._lc = state_dict['mask_params'][-1]
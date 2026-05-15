import torch
import numpy as np
from .gaussian3d import Gaussian3D  # Assumindo que está no mesmo pacote

class OPFGaussian3D(Gaussian3D):
    def __init__(self, cfg, log, work_dir, debug=False):
        super().__init__(cfg, log, work_dir, debug)
        self._weights = {
            "xyz": torch.tensor(1.0),
            "f_dc": torch.tensor(1.0),
            "f_rest": torch.tensor(1.0),
            "scaling": torch.tensor(1.0),
            "rotation": torch.tensor(1.0),
            "opacity": torch.tensor(1.0)
        }
        self._lc = torch.empty(0)

    def update_opf_weights(self):
        property_groups = {
            "xyz": self._xyz,
            "f_dc": self._features_dc.flatten(1),
            "f_rest": self._features_rest.flatten(1),
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity
        }
        precisions = {}
        with torch.no_grad():
            for name, data in property_groups.items():
                if data.shape[0] < 2:
                    continue
                var_vector = torch.var(data, dim=0)
                total_variance = torch.sum(var_vector)
                precisions[name] = 1.0 / (total_variance + 1e-7)
            total_p = sum(precisions.values())
            for name in precisions:
                self._weights[name] = precisions[name] / total_p

    def train_step(self, cur_iter, total_iter, viewpoint, render, update_freq=10):
        super().train_step(cur_iter, total_iter, viewpoint, render)
        if cur_iter % update_freq == 0:
            self.update_opf_weights()
            # Exemplo: self._lc = sum(self._weights[k] * log_prob_k)

    @property
    def get_opf_weights(self):
        return self._weights

    def save_point_cloud(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()        
        weights = np.concatenate([w.detach().cpu().numpy().ravel() for w in self._weights.values()])

        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i)) 
        for weights in weights.keys():
            l.append('w_{}'.format(weights))
        
        dtype_full = [(attribute, 'f4') for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, weights), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
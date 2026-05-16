import torch
import numpy as np
from plyfile import PlyData, PlyElement
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

    def state_dict(self):
        state_dict = super().state_dict()
        opf_weights = {k: float(v.detach().cpu().item()) for k, v in self._weights.items()}
        state_dict['hyperparams'] = state_dict['hyperparams'] + (opf_weights,)  # type: ignore[assignment]  
        return state_dict

    def load_state_dict(self, state_dict, optim_cfg):
        saved_weights = None
        hyperparams = state_dict.get('hyperparams', ())
        if len(hyperparams) > 2 and isinstance(hyperparams[-1], dict):
            saved_weights = hyperparams[-1]
            state_dict = dict(state_dict)
            state_dict['hyperparams'] = hyperparams[:-1]

        super().load_state_dict(state_dict, optim_cfg)

        if saved_weights is not None:
            device = self._xyz.device if hasattr(self._xyz, 'device') else torch.device('cpu')
            self._weights = {
                name: torch.tensor(float(saved_weights.get(name, self._weights[name].item())), device=device)
                for name in self._weights
            }

    def optimize_step(self, cur_iter, _total_iter, train_cfg, extra_info):
        total_iter = _total_iter
        super().optimize_step(cur_iter, total_iter, train_cfg, extra_info)
        interval = getattr(train_cfg, 'weight_precision_interval', 10)
        if interval > 0 and cur_iter % interval == 0:
            self.update_opf_weights()

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_weights=None):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

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
        
        dtype_full = [(attribute, 'f4') for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
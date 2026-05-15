from .gaussian3d import Gaussian3D
from .masked_gaussian3d import MaskedGaussian3D
from .opf_gaussian3d import OPFGaussian3D
from .uv_map_gaussian3d import UVMapGaussian3D
from .texture_gaussian3d import TextureGaussian3D
type2model = dict(
    Gaussian3D=Gaussian3D,
    MaskedGaussian3D=MaskedGaussian3D,
    OPFGaussian3D=OPFGaussian3D,
    UVMapGaussian3D=UVMapGaussian3D,
    TextureGaussian3D=TextureGaussian3D,
)

def create_model(cfg, *args, **kargs):
    return type2model[cfg.type](cfg, *args, **kargs)
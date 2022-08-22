from scipy.spatial.transform import Rotation
from isaacgym import gymapi
from enum import Enum
import torch

class QuaternionOrder(Enum):
    XYZW = 0
    WXYZ = 1

@torch.jit.script
def quaternion_to_rotation_matrix(
        quaternion: torch.Tensor
    ) -> torch.Tensor:
    
    x, y, z, w = torch.chunk(quaternion, chunks=4, dim=quaternion.dim()-1)
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z

    matrix = torch.stack([
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ], dim=quaternion.dim()-1,
    ).view(quaternion.size(0), 3, 3)

    return matrix

@torch.jit.script
def quaternion_to_euler(
        quaternion: torch.Tensor
    ) -> torch.Tensor:

    x, y, z, w = torch.chunk(quaternion, chunks=4, dim=-1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    ).view(-1, 3)

    if len(euler_angles.shape) == 1:
        euler_angles = torch.squeeze(euler_angles, dim=0)
    return euler_angles

# @torch.jit.script
def rotation_matrix_to_euler(
        rotation_matrix: torch.Tensor
    ) -> torch.Tensor:

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix.reshape(-1, 9), chunks=9, dim=-1)

    # compute the euler angles
    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(m21, m22),
            torch.asin(-m20),
            torch.atan2(m10, m00),
        ),
        dim=-1,
    ).view(-1, 3)

    return euler_angles

# @torch.jit.script  
def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    r, p, y = torch.chunk(euler, chunks=3, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack((qx, qy, qz, qw), dim=euler.dim()-1)
    
    return quaternion

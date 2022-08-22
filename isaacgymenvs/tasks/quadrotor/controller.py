from typing import Tuple
from .utils import *


class DSLPIDControl:
    """
    TODO: @ Botian: completely test the functionality of this controller...
    """
    def __init__(self, n, sim_params: gymapi.SimParams, kf, device="cpu") -> None:
        self.device = device
        self.P_COEFF_FOR = torch.tensor([.4, .4, 1.25], device=device)
        self.I_COEFF_FOR = torch.tensor([.05, .05, .05], device=device)
        self.D_COEFF_FOR = torch.tensor([.2, .2, .5], device=device)
        self.P_COEFF_TOR = torch.tensor([70000., 70000., 60000.], device=device)
        self.I_COEFF_TOR = torch.tensor([.0, .0, 500.], device=device)
        self.D_COEFF_TOR = torch.tensor([20000., 20000., 12000.], device=device)
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000.
        self.MAX_PWM = 65535.
        self.GRAVITY = abs(sim_params.gravity.z) * 0.027
        self.MIXER_MATRIX = torch.tensor([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ], device=device)
        self.KF = kf
        self.N = n

        self.last_pos_e = torch.zeros((n, 3), device=self.device)
        self.integral_pos_e = torch.zeros((n, 3), device=self.device)
        self.last_rpy = torch.zeros((n, 3), device=self.device)
        self.last_rpy_e = torch.zeros((n, 3), device=self.device)
        self.integral_rpy_e = torch.zeros((n, 3), device=self.device)

    def reset_idx(self, ids: torch.Tensor, num_envs: int):
        self.last_pos_e.view(num_envs, -1)[ids] = 0
        self.integral_pos_e.view(num_envs, -1)[ids] = 0
        self.last_rpy.view(num_envs, -1)[ids] = 0
        self.last_rpy_e.view(num_envs, -1)[ids] = 0
        self.integral_rpy_e.view(num_envs, -1)[ids] = 0

    def compute_control(self, control_timestep, 
            cur_pos, cur_quat, cur_vel, cur_ang_vel,
            target_pos, target_rpy, target_vel, target_rpy_rates):
        # thrust, computed_target_rotation, pos_e = self._position_control(
        #     control_timestep,
        #     cur_pos,
        #     cur_quat,
        #     cur_vel,
        #     target_pos,
        #     target_rpy,
        #     target_vel
        #     )
        # rpm = self._attitude_control(
        #     control_timestep,
        #     thrust,
        #     cur_quat,
        #     computed_target_rotation,
        #     target_rpy_rates
        #     )
        rpms, self.last_rpy[:], self.integral_pos_e[:], self.integral_rpy_e[:] = compute_control(
            control_timestep, 
            cur_pos, cur_quat, cur_vel, cur_ang_vel,
            target_pos, target_rpy, target_vel, target_rpy_rates, 
            self.last_rpy, self.integral_rpy_e, self.integral_pos_e,
            self.MIXER_MATRIX, self.PWM2RPM_SCALE, self.PWM2RPM_CONST,
            self.P_COEFF_FOR, self.I_COEFF_FOR, self.D_COEFF_FOR,
            self.P_COEFF_TOR, self.I_COEFF_TOR, self.D_COEFF_TOR,
            self.GRAVITY, self.MIN_PWM, self.MAX_PWM, self.KF
        )
        return rpms

    def _position_control(self, 
            control_timestep: float,
            cur_pos: torch.Tensor,
            cur_quat: torch.Tensor,
            cur_vel: torch.Tensor,
            target_pos: torch.Tensor,
            target_rpy: torch.Tensor,
            target_vel: torch.Tensor):
        
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        cur_rotation = quaternion_to_rotation_matrix(cur_quat) # (*, 3, 3)
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = torch.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[..., 2] = torch.clip(self.integral_pos_e[..., 2], -0.15, .15)

        target_thrust = self.P_COEFF_FOR * pos_e \
            + self.I_COEFF_FOR * self.integral_pos_e \
            + self.D_COEFF_FOR * vel_e \
            + torch.tensor([0, 0, self.GRAVITY], device=self.device) # (*, 3)
        scalar_thrust = torch.relu(target_thrust.view(-1, 1, 3) @ cur_rotation[..., :, 2].view(-1, 3, 1)).view(-1, 1)
        thrust = (torch.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        target_z_ax = target_thrust / torch.linalg.norm(target_thrust, axis=-1, keepdims=True)
        target_x_c = torch.stack([
            torch.cos(target_rpy[..., 2]), torch.sin(target_rpy[..., 2]), torch.zeros_like(target_rpy[..., 2])], dim=-1)
        target_y_ax = torch.cross(target_z_ax, target_x_c) / torch.linalg.norm(torch.cross(target_z_ax, target_x_c), axis=-1, keepdims=True)
        target_x_ax = torch.cross(target_y_ax, target_z_ax)
        target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
        return thrust, target_rotation, pos_e
    
    def _attitude_control(self,
            control_timestep: float,
            thrust: torch.Tensor,
            cur_quat: torch.Tensor,
            target_rotation: torch.Tensor,
            target_rpy_rates: torch.Tensor):
        cur_rotation = quaternion_to_rotation_matrix(cur_quat)
        cur_rpy = quaternion_to_euler(cur_quat)

        rot_matrix_e = target_rotation.transpose(-1, -2) @ cur_rotation - cur_rotation.transpose(-1, -2) @ target_rotation

        rot_e = torch.stack([rot_matrix_e[..., 2, 1], rot_matrix_e[..., 0, 2], rot_matrix_e[..., 1, 0]], dim=-1)
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = torch.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[..., 0:2] = torch.clip(self.integral_rpy_e[..., 0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - self.P_COEFF_TOR * rot_e \
                         + self.D_COEFF_TOR * rpy_rates_e \
                         + self.I_COEFF_TOR * self.integral_rpy_e
        
        target_torques = torch.clip(target_torques, -3200, 3200)
        pwm = thrust + (self.MIXER_MATRIX @ target_torques.T).T
        pwm = torch.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

from torch import Tensor

@torch.jit.script
def compute_control(
    control_timestep: float, 
    cur_pos: Tensor, cur_quat: Tensor, cur_vel: Tensor, cur_ang_vel: Tensor,
    target_pos: Tensor, target_rpy: Tensor, target_vel: Tensor, target_rpy_rates: Tensor,
    last_rpy: Tensor, integral_rpy_e: Tensor, integral_pos_e: Tensor,
    MIXER_MATRIX: Tensor, PWM2RPM_SCALE: Tensor, PWM2RPM_CONST: Tensor,
    P_COEFF_FOR: Tensor, I_COEFF_FOR: Tensor, D_COEFF_FOR: Tensor,
    P_COEFF_TOR: Tensor, I_COEFF_TOR: Tensor, D_COEFF_TOR: Tensor,
    GRAVITY: float, MIN_PWM: float, MAX_PWM: float, KF: float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # position control
    pos_e = target_pos - cur_pos
    vel_e = target_vel - cur_vel

    cur_rotation = quaternion_to_rotation_matrix(cur_quat) # (*, 3, 3)

    integral_pos_e = integral_pos_e + pos_e*control_timestep
    integral_pos_e = torch.clip(integral_pos_e, -2., 2.)
    integral_pos_e[..., 2] = torch.clip(integral_pos_e[..., 2], -0.15, .15)

    target_thrust = P_COEFF_FOR * pos_e \
        + I_COEFF_FOR * integral_pos_e \
        + D_COEFF_FOR * vel_e
    target_thrust[2] += GRAVITY

    scalar_thrust = torch.relu(target_thrust.view(-1, 1, 3) @ cur_rotation[..., :, 2].view(-1, 3, 1)).view(-1, 1)
    thrust = (torch.sqrt(scalar_thrust / (4 * KF)) - PWM2RPM_CONST) / PWM2RPM_SCALE
    target_z_ax = target_thrust / torch.norm(target_thrust, dim=-1, keepdim=True)
    target_x_c = torch.stack([
        torch.cos(target_rpy[..., 2]), torch.sin(target_rpy[..., 2]), torch.zeros_like(target_rpy[..., 2])], dim=-1)
    target_y_ax = torch.cross(target_z_ax, target_x_c) / torch.norm(torch.cross(target_z_ax, target_x_c), dim=-1, keepdim=True)
    target_x_ax = torch.cross(target_y_ax, target_z_ax)
    target_rotation = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=-1)
    
    # attitude control
    cur_rpy = quaternion_to_euler(cur_quat)

    rot_matrix_e = target_rotation.transpose(-1, -2) @ cur_rotation - cur_rotation.transpose(-1, -2) @ target_rotation

    rot_e = torch.stack([rot_matrix_e[..., 2, 1], rot_matrix_e[..., 0, 2], rot_matrix_e[..., 1, 0]], dim=-1)
    rpy_rates_e = target_rpy_rates - (cur_rpy - last_rpy) / control_timestep

    integral_rpy_e = integral_rpy_e - rot_e*control_timestep
    integral_rpy_e = torch.clip(integral_rpy_e, -1500., 1500.)
    integral_rpy_e[..., 0:2] = torch.clip(integral_rpy_e[..., 0:2], -1., 1.)
    #### PID target torques ####################################
    target_torques = - P_COEFF_TOR * rot_e \
                        + D_COEFF_TOR * rpy_rates_e \
                        + I_COEFF_TOR * integral_rpy_e
    
    target_torques = torch.clip(target_torques, -3200, 3200)
    pwm = thrust + (MIXER_MATRIX @ target_torques.T).T
    pwm = torch.clip(pwm, MIN_PWM, MAX_PWM)
    rpms = PWM2RPM_SCALE * pwm + PWM2RPM_CONST
    return (
        rpms,
        cur_rpy,
        integral_pos_e,
        integral_rpy_e,
    )


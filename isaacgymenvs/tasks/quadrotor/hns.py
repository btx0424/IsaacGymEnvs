from .quadrotor import QuadrotorBase

class FixedPrey(QuadrotorBase):
    def reset(self):
        return super().reset()
        
    def compute_reward(self):
        return super().compute_reward()
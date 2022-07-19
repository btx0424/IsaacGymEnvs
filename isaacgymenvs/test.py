import hydra
from omegaconf import OmegaConf
from isaacgym import gymutil
from isaacgymenvs.tasks.quadrotor import Quadrotor

import torch
import time
import time

@hydra.main(config_path="./cfg", config_name="config")
def test(config):
    OmegaConf.set_struct(config, False)
    env = Quadrotor(
        config.task, config.rl_device, config.sim_device, 
        config.graphics_device_id, config.headless)
    print(env)
    env.reset()
    
    steps = 0
    time_start = time.perf_counter()

    # actions = torch.tensor([[
    #     [0.0, 0.3, 1.0],
    #     [0.1, 0.0, 0.2],
    #     [-.5, 0.0, 1.2],
    #     [-.1, 0.6, 0.8]
    #     ]], device=env.device).view(4, 1, 3)
    actions = torch.rand((env.num_environments, env.num_agents, 3), device=env.device)
    
    while True:
        obs, reward, done, info = env.step(actions)
        env.render()
        steps += env.num_envs
        # if done.all():
        #     print(env.root_positions)
        #     print(torch.norm(env.root_positions - actions, dim=-1).mean())
        # if (time.perf_counter() - time_start) > 10:
        #     break
        env_done = done.all(-1)
        if env_done.any():
            print(info["cum_rew"][env_done].mean())

    print("fps:", steps / (time.perf_counter() - time_start))

if __name__ == "__main__":
    test()
    
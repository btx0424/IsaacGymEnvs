import hydra
from omegaconf import OmegaConf
from isaacgym import gymutil
from isaacgymenvs.tasks.quadrotor import Quadrotor

import torch
import time

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_path="./cfg", config_name="config")
def test(config):
    OmegaConf.set_struct(config, False)
    env = Quadrotor(config.task, config.sim_device, config.graphics_device_id, config.headless)
    env.reset()
    
    steps = 0
    time_start = time.perf_counter()
    while True:
        # actions = torch.tensor([
        #     [0.0, 0.3, 1.0],
        #     [0.1, 0.0, 0.2],
        #     [-.5, 0.0, 1.2],
        #     [-.1, 0.6, 0.8]
        # ], device=env.device).view(4, 1, 3)
        actions = torch.rand((env.num_environments, env.num_agents, 3), device=env.device)
        obs, reward, done, info = env.step(actions)
        # env.render()
        steps += env.num_envs
        # if done.all():
        #     print(env.root_positions)
        #     print(torch.norm(env.root_positions - actions, dim=-1).mean())
        if steps > 5e6:
            break

    print("fps:", steps / (time.perf_counter() - time_start))

if __name__ == "__main__":
    test()
    
import hydra
import isaacgym
import torch
import wandb
import setproctitle
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from omegaconf import OmegaConf
from onpolicy.runner.shared.drone_runner_coop import DroneRunner

def create_envs(cfg) -> MultiAgentVecTask:
    task_config = cfg.task
    env = isaacgym_task_map[cfg.task_name](
        cfg=task_config,
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=cfg.capture_video,
        force_render=cfg.force_render,
    )
    return env

@hydra.main(config_name="mappo", config_path="./cfg")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    env: MultiAgentVecTask = create_envs(cfg)
    env.obs_space = [env.obs_space] * env.num_agents
    env.act_space = [env.act_space] * env.num_agents
    env.share_observation_space = env.obs_space

    config = {
        "all_args": cfg.params,
        "envs": env,
        "device": "cuda",
        "num_agents":cfg.task.env.numAgents,
    }
    runner = DroneRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
from argparse import Namespace
import logging
import gym
import hydra
import isaacgym
from isaacgymenvs.utils.wrappers import MultiAgentRecordVideo
import torch
import os
import wandb
import datetime
import setproctitle
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.tasks.base.vec_task import MultiAgentVecTask
from omegaconf import OmegaConf
from isaacgymenvs.learning.mappo.runner.shared.drone_runner_coop import DroneRunner

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

@hydra.main(config_name="mappo", config_path="./cfg", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_name = f"{cfg.wandb_name}/mappo_{time_str}"
    setproctitle.setproctitle(run_name)

    run_id = cfg.resume_id or wandb.util.generate_id()
    run = wandb.init(
        id=run_id,
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        monitor_gym=True,
        name=run_name,
        resume="allow",
    )
        
    envs: MultiAgentVecTask = create_envs(cfg)
    if cfg.capture_video:
        envs.is_vector_env = True
        envs = MultiAgentRecordVideo(
            envs,
            f"videos/{run_name}",
            episode_trigger=lambda ep_id: ep_id % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )
    print(envs)
    cfg = wandb.config
    config = {
        "cfg": cfg,
        "all_args": Namespace(**cfg.params),
        "envs": envs,
    }
    runner = DroneRunner(config)
    if run.resumed:
        logging.info(f"Resuming run {run.id}")
        wandb.restore("checkpoint.pt")
        runner.restore()
    runner.run()
        
if __name__ == "__main__":
    main()
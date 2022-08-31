from argparse import Namespace
import logging
from pprint import pprint
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
from isaacgymenvs.learning.mappo.runner.shared.drone_runner import DroneRunner

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
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{cfg.wandb_name}/{time_str}"
    if cfg.capture_video: cfg.headless = False
    setproctitle.setproctitle(run_name)

    if cfg.resume_id is not None:
        if cfg.run_path is not None:
            logging.warning(f"run_path and resume_id are both set, using resume_id {cfg.resume_id}")
        run = wandb.init(
            id=cfg.run_id,
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            monitor_gym=True,
            name=run_name,
            resume="allow",
            job_type="eval"
        )
        wandb.restore("checkpoint.pt")
    elif cfg.run_path is not None:
        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            monitor_gym=True,
            name=run_name,
            resume="allow",
            job_type="eval"
        )
        wandb.restore("checkpoint_best.pt", run_path=cfg.run_path)
    else:
        raise ValueError("Must provide a resume_id or run_path to eval!")
    
    run.config.update(OmegaConf.to_container(cfg, resolve=True))

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
    runner.restore(tag="best")
    runner.eval(cfg.eval_episodes, log=False, verbose=True)
        
if __name__ == "__main__":
    main()
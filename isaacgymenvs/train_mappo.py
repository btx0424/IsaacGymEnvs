import gym
import hydra
import isaacgym
from isaacgymenvs.utils.wrappers import MultiAgentRecordVideo
import torch
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

@hydra.main(config_name="mappo", config_path="./cfg")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}/mappo_{time_str}"
    setproctitle.setproctitle(run_name)
    
    envs: MultiAgentVecTask = create_envs(cfg)
    if cfg.capture_video:
        envs.is_vector_env = True
        envs = MultiAgentRecordVideo(
            envs,
            f"videos/{run_name}",
            episode_trigger=lambda ep_id: ep_id % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )
    print(OmegaConf.to_yaml(cfg))
    print(envs)

    config = {
        "cfg": cfg,
        "all_args": cfg.params,
        "envs": envs,
        "device": "cuda",
    }

    run = wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        monitor_gym=True,
        name=run_name,
        resume="allow",
    )

    runner = DroneRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
import gym
import hydra
import isaacgym
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

    envs: MultiAgentVecTask = create_envs(cfg)
    if cfg.capture_video:
        envs.is_vector_env = True
        class MultiAgentRecordVideo(gym.wrappers.RecordVideo):

            def step(self, action):
                observations, rewards, dones, infos = super().step(action)
                
                # increment steps and episodes
                self.step_id += 1

                env_done = dones.reshape(self.num_envs, self.num_agents)[0].all()
                if env_done:
                    self.episode_id += 1

                if self.recording:
                    self.video_recorder.capture_frame()
                    self.recorded_frames += 1
                    if self.video_length > 0:
                        if self.recorded_frames > self.video_length:
                            self.close_video_recorder()
                    elif env_done:
                        self.close_video_recorder()

                elif self._video_enabled():
                    self.start_video_recorder()

                return observations, rewards, dones, infos

        envs = MultiAgentRecordVideo(
            envs,
            f"videos/{run_name}",
            episode_trigger=lambda ep_id: ep_id % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )
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
        config=cfg,
        sync_tensorboard=True,
        name=run_name,
        resume="allow",
    )

    runner = DroneRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
import gym
import os
from gym.wrappers.monitoring import video_recorder

class MultiAgentRecordVideo(gym.wrappers.RecordVideo):

    def step(self, action):
        observations, rewards, dones, infos = self.env.step(action)

        # increment steps and episodes
        self.step_id += 1

        env_done = dones.reshape(self.num_envs, self.num_agents)[0].all()
        if env_done:
            self.episode_id += 1

        if self.recording:
            if (self.recorded_frames-1) % 2 ==0: 
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
    
    def close_video_recorder(self) -> None:
        super().close_video_recorder()
        self.env.enable_viewer_sync = False
    
    def start_video_recorder(self) -> None:
        self.close_video_recorder()
        
        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={"step_id": self.step_id, "episode_id": self.episode_id},
        )

        self.env.enable_viewer_sync = True
        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

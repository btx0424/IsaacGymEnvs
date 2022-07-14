# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=1 max_iterations=2000 \
#     train=QuadrotorPPO_RNN num_envs=16384 

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=1 max_iterations=2000 \
#     train=QuadrotorPPO num_envs=16384 

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=1 max_iterations=2000 \
#     train=QuadrotorPPO_RNN num_envs=16384 task.env.actType=pid_waypoint

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=1 max_iterations=2000 \
#     train=QuadrotorPPO num_envs=16384 task.env.actType=pid_waypoint

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=2 max_iterations=2000 \
#     train=QuadrotorPPO_RNN num_envs=16384 

python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
    headless=False num_agents=2 max_iterations=2000 \
    train=QuadrotorPPO num_envs=16384 capture_video=True

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=2 max_iterations=2000 \
#     train=QuadrotorPPO_RNN num_envs=16384 task.env.actType=pid_waypoint

# python train_drone.py task=Quadrotor wandb_activate=True wandb_entity=btx0424 \
#     headless=True num_agents=2 max_iterations=2000 \
#     train=QuadrotorPPO num_envs=16384 task.env.actType=pid_waypoint capture_video=True
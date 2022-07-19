python train_mappo.py capture_video=True num_envs=16384 params.num_steps=32\
    params.ppo_epoch=5 params.num_mini_batch=16 

python train_mappo.py capture_video=True num_envs=16384 params.num_steps=32\
    params.ppo_epoch=5 params.num_mini_batch=8 

python train_mappo.py capture_video=True num_envs=16384 params.num_steps=32\
    params.ppo_epoch=10 params.num_mini_batch=8 

python train_mappo.py capture_video=True num_envs=16384 params.num_steps=32\
    params.ppo_epoch=5 params.num_mini_batch=1 

python train_mappo.py capture_video=True num_envs=16384 params.num_steps=32\
    params.ppo_epoch=10 params.num_mini_batch=1 
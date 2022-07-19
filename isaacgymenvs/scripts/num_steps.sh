python train_mappo.py capture_video=True \
    params.num_steps=64 num_envs=8192

python train_mappo.py capture_video=True \
    params.num_steps=32 num_envs=16384

python train_mappo.py capture_video=True \
    params.num_steps=16 num_envs=16384

python train_mappo.py capture_video=True \
    params.num_steps=16 num_envs=32768

python train_mappo.py capture_video=True \
    params.num_steps=16 num_envs=8192

python train_mappo.py capture_video=True \
    params.num_steps=8 num_envs=65536
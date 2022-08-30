from collections import defaultdict
import torch
import numpy as np
from typing import Callable, Dict, List, Mapping, Union


class TensorDict(Dict[str, torch.Tensor]):
    def reshape(self, *shape: int):
        return TensorDict({key: value.reshape(*shape) for key, value in self.items()})

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        return TensorDict({key: value.flatten(start_dim, end_dim) for key, value in self.items()})
    
    def unflatten(self, dim, sizes):
        return TensorDict({key: value.unflatten(dim, sizes) for key, value in self.items()})

    def unsqueeze(self, dim: int):
        return TensorDict({key: value.unsqueeze(dim) for key, value in self.items()})

    def __getitem__(self, k):
        if isinstance(k, str):
            return super().get(k)
        else:
            return TensorDict({key: value[k] for key, value in self.items()})
    
    def filter(self, condition):
        return TensorDict({k: v for k, v in self.items() if condition(k, v)})
    
    def select(self, *keys: str):
        return TensorDict({k: self.get(k) for k in keys if self.get(k) is not None})

    def step(self):
        keys = [key for key in self.keys() if key.startswith("next_")]
        new_keys = [key[5:] for key in keys]
        return TensorDict({new_key: self[key] for key, new_key in zip(keys, new_keys)}) 
    
    def drop(self, *keys: str):
        return TensorDict({k: self[k] for k in self.keys() if k not in keys})

    def rename(self, **mapping):
        return TensorDict({mapping[key]if key in mapping.keys() else key: value for key, value in self.items()})

    def cpu(self):
        return TensorDict({key: value.cpu() for key, value in self.items()})

    def numpy(self) -> Dict[str, np.ndarray]:
        return {key: value.numpy() for key, value in self.items()}

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        return self
    
    def set(self, index=None, **kwargs):
        if index is not None:
            for k, v in kwargs:
                self[k][index] = v
        else:
            for k, v in kwargs:
                self[k][:] = v
        return self

    def mean(self, dim:int=None):
        return TensorDict({key: value.mean(dim) for key, value in self.items()})


class LazyRolloutBuffer(TensorDict):
    def __init__(self, size=64, stack_dim=0):
        super().__init__()
        self.size = size
        self.stack_dim = stack_dim
        self._step = 0
        self._initiaized = False

    def insert(self, dict: Dict[str, torch.Tensor], step=None):
        if not self._initiaized:
            for k, v in dict.items():
                if step:
                    assert self.stack_dim == 0
                    base_shape = v.shape[1:]
                else: 
                    base_shape = v.shape
                size = (*base_shape[:self.stack_dim], self.size, *base_shape[self.stack_dim:])
                self[k] = torch.zeros(size, dtype=v.dtype, device=v.device)
            self._initiaized = True
        else:
            _step = self._step % self.size
            idx = _step if step is None else torch.arange(_step, _step+step)%self.size
            for k, v in dict.items():
                self[k][idx] = v
        self._step += step or 1
    
    def __len__(self) -> int:
        return min(self._step, self.size)

    def update(self, dict: TensorDict):
        if self._initiaized:
            for k, v in dict.items():
                assert v.size(self.stack_dim) == self.size
                self[k] = v
        else:
            raise RuntimeError()
        return self

def group_by_agent(tensordict: TensorDict):
    grouped = defaultdict(TensorDict)
    others = TensorDict()
    for k, v in tensordict.items():
        k = k.split("@")
        if len(k) > 1:
            k, agent = k
            grouped[agent][k] = v
        else:
            others[k[0]] = v
    return grouped, others
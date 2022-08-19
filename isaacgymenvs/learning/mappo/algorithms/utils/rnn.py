
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.dim() == 2:
            x.unsqueeze_(0)
        assert hxs.dim() == 3, 'Expected hidden states to have dimension of 3, but got {}'.format(hxs.dim())
        T, N = x.shape[:2]
        
        # Let's figure out which steps in the sequence have a zero for any agent
        # We will always assume t=0 has a zero in it as that makes the logic cleaner
        has_zeros = ((masks.squeeze()[1:] == 0.0)
                        .any(dim=-1)
                        .nonzero()
                        .squeeze()
                        .cpu())

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # Deal with scalar
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [T]
        outputs = []
        for i in range(len(has_zeros) - 1):
            # We can now process steps that don't have any zeros in masks together!
            # This is much faster
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]               
            temp = hxs * masks[start_idx].unsqueeze(0)
            rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
            outputs.append(rnn_scores)

        outputs = x + torch.cat(outputs, dim=0)
        outputs = self.norm(outputs.view(T * N, -1)).view(T, N, -1)

        return outputs, hxs

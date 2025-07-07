import torch
import numpy as np


def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_masks(src, tgt, pad_idx=0):
    src_mask = create_padding_mask(src, pad_idx)
    
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)
    tgt_len = tgt.shape[1]
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
    
    return src_mask, tgt_mask


class NoamOpt:
    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model, d_model=512, factor=2, warmup=4000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return NoamOpt(d_model, factor, warmup, optimizer)
"""Minimal source-pinned utilities for QuIP# Hessian collection.

Copied in behavior from Cornell-RelaxML/quip-sharp commit
1d8f873e9a2a8b86b12bb1064c312c5689b77d98:
- lib/utils/data_utils.py: sym_to_flat, register_H_hook, wrap_tokenizer, sample_rp1t
- lib/utils/misc.py: clean

The full QuIP# lib.utils import pulls codebook/inference machinery that is irrelevant
to Hessian collection. This shim preserves the collector math, including `mu`, while
avoiding unrelated CUDA extensions.
"""
from __future__ import annotations

import gc
import multiprocessing as mp

import torch
from datasets import load_dataset


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


def register_H_hook(module, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def wrap_tokenizer(tokenizer, x, ctx_size):
    return tokenizer(x,
                     return_tensors='pt',
                     truncation=True,
                     padding=True,
                     max_length=ctx_size)


def sample_rp1t(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                           split='train')
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    if nproc > 1:
        p = mp.Pool(nproc)
        try:
            while saved < size:
                seqs = [(tokenizer, dataset[torch.randint(len(dataset),
                                                          (size, ))]['text'],
                         ctx_size) for _ in range(nproc)]
                tokens = p.starmap(wrap_tokenizer, seqs)
                for i in range(len(tokens)):
                    lens = tokens[i].attention_mask.sum(dim=-1)
                    good = torch.where(lens == ctx_size)[0]
                    if len(good) > 0:
                        if saved + len(good) > size:
                            good = good[:size - saved]
                        devset[saved:saved + len(good)] = tokens[i].input_ids[good]
                        saved += len(good)
                        print(saved)
        finally:
            p.close()
            p.join()
    else:
        while saved < size:
            tokens = tokenizer(dataset[torch.randint(len(dataset),
                                                     (size, ))]['text'],
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=ctx_size)
            lens = tokens.attention_mask.sum(dim=-1)
            good = torch.where(lens == ctx_size)[0]
            if len(good) > 0:
                if saved + len(good) > size:
                    good = good[:size - saved]
                devset[saved:saved + len(good)] = tokens.input_ids[good]
                saved += len(good)
    return devset

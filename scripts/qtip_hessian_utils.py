"""Minimal utilities required by QTIP's input_hessian_llama.py.

The functions below are intentionally copied from the pinned QTIP commit
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`:
- lib/utils/data_utils.py: sym_to_flat, register_input_H_hook,
  wrap_tokenizer, sample_rp1t_concat
- lib/utils/misc.py: clean

Why this shim exists:
`from lib import utils` imports QTIP's entire utility package, which in turn imports
QTIP inference/codebook CUDA kernels (`qtip_kernels`) and other dependencies that
are irrelevant to Hessian collection. The upstream Hessian collector only calls the
functions reproduced here. Keeping this tiny shim lets us execute the upstream
collector body while avoiding unrelated QTIP inference extensions.
"""
from __future__ import annotations

import gc
import multiprocessing as mp

import torch
from datasets import load_dataset


# Copied verbatim in behavior from QTIP lib/utils/misc.py.
def clean():
    gc.collect()
    torch.cuda.empty_cache()


# Copied verbatim in behavior from QTIP lib/utils/data_utils.py.
def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


# Copied verbatim in behavior from QTIP lib/utils/data_utils.py.
def register_input_H_hook(module, save_pfx, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, ct, hook
        save_path = f"{save_pfx}_{device}.pt"
        torch.save({'H': H, 'n': H.shape[0], 'ct': ct}, save_path)
        del H, ct
        hook.remove()
        del hook
        clean()

    return done


# Copied verbatim in behavior from QTIP lib/utils/data_utils.py.
def wrap_tokenizer(tokenizer, x, ctx_size, truncate=True):
    return tokenizer(x,
                     return_tensors='pt',
                     truncation=truncate,
                     padding=True,
                     max_length=ctx_size)


# Copied verbatim in behavior from QTIP lib/utils/data_utils.py.
def sample_rp1t_concat(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                           split='train')
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    concat = []
    p = mp.Pool(nproc)
    try:
        while len(concat) < ctx_size * size:
            seqs = [(tokenizer, dataset[torch.randint(len(dataset),
                                                      (128, ))]['text'], -1, False)
                    for _ in range(nproc)]
            tokens = p.starmap(wrap_tokenizer, seqs)
            for i in range(len(tokens)):
                lens = tokens[i].attention_mask.sum(dim=-1)
                for j in range(len(tokens[i].input_ids)):
                    concat += tokens[i].input_ids[j][:lens[j]]
            print(len(concat), ctx_size * size)
    finally:
        p.close()
        p.join()
    concat = torch.tensor(concat)[:ctx_size * size]
    return concat.reshape(size, ctx_size).contiguous()

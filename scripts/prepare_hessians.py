from __future__ import annotations

import os
from pathlib import Path

from .common import ARTIFACTS, MODEL_ID, ROOT, UPSTREAM, clean_env, run
from .quantize_vptq import validate_hessian_dir
from .setup_env import env_python


def main() -> Path:
    explicit = os.environ.get("VPTQ_HESSIAN_DIR")
    if explicit:
        path = Path(explicit).resolve()
        validate_hessian_dir(path)
        print(f"[HESSIAN] using provided directory: {path}")
        return path

    out = ARTIFACTS / "hessians_llama2_7b_fingerprinted"
    try:
        validate_hessian_dir(out)
        print(f"[SKIP] Hessians already complete: {out}")
        return out
    except RuntimeError:
        pass

    out.mkdir(parents=True, exist_ok=True)
    qtip = UPSTREAM / "qtip"
    script = qtip / "quantize_llama" / "input_hessian_llama.py"
    if not script.exists():
        raise RuntimeError("Pinned QTIP Hessian collector missing")

    # Colab-practical defaults. Set HESSIAN_DEVSET_SIZE=8192 to reproduce the
    # upstream collector's original full default; 512 is much faster for attack screening.
    devset = int(os.environ.get("HESSIAN_DEVSET_SIZE", "512"))
    ctx = int(os.environ.get("HESSIAN_CTX_SIZE", "4096"))
    batch = int(os.environ.get("HESSIAN_BATCH_SIZE", "1"))
    large_batch = int(os.environ.get("HESSIAN_LARGE_BATCH_SIZE", str(devset)))
    sample_proc = int(os.environ.get("HESSIAN_SAMPLE_PROC", "4"))

    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # torchrun initializes the single-GPU NCCL process group expected by upstream.
    cmd = [
        env_python(), "-m", "torch.distributed.run", "--nproc_per_node=1",
        str(script),
        "--base_model", MODEL_ID,
        "--save_path", str(out),
        "--batch_size", str(batch),
        "--large_batch_size", str(large_batch),
        "--devset_size", str(devset),
        "--ctx_size", str(ctx),
        "--sample_proc", str(sample_proc),
    ]
    run(cmd, cwd=qtip, env=env)
    validate_hessian_dir(out)
    print(f"[OK] generated upstream-format Hessians: {out}")
    return out


if __name__ == "__main__":
    print(main())

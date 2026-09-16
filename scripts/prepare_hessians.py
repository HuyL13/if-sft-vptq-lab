from __future__ import annotations

import os
from pathlib import Path

from .common import ARTIFACTS, MODEL_ID, ROOT, UPSTREAM, clean_env, run
from .quantize_vptq import validate_hessian_dir
from .setup_env import env_python


def build_runtime_collector(qtip: Path) -> Path:
    """Create a runtime copy of QTIP's collector with exactly one import replaced.

    The upstream collector body remains unchanged. We replace only
    `from lib import utils` with our source-pinned minimal shim because QTIP's full
    utils package imports qtip_kernels and other inference-only dependencies that
    are unrelated to Hessian collection and are not compatible with current Colab.
    """
    upstream_script = qtip / "quantize_llama" / "input_hessian_llama.py"
    if not upstream_script.exists():
        raise RuntimeError("Pinned QTIP Hessian collector missing")

    text = upstream_script.read_text(encoding="utf-8")
    old = "from lib import utils"
    new = "from scripts import qtip_hessian_utils as utils"
    if old not in text:
        raise RuntimeError("Pinned QTIP collector changed unexpectedly; expected import not found")

    runtime_dir = ARTIFACTS / "runtime_patches"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_script = runtime_dir / "input_hessian_llama.py"
    runtime_script.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[HESSIAN] runtime collector: {runtime_script}")
    print("[HESSIAN] QTIP patch: only `from lib import utils` -> source-pinned minimal Hessian shim")
    return runtime_script


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
    runtime_script = build_runtime_collector(qtip)

    # Colab-practical defaults. Set HESSIAN_DEVSET_SIZE=8192 to reproduce the
    # upstream collector's original full default. 512 is used for attack screening.
    devset = int(os.environ.get("HESSIAN_DEVSET_SIZE", "512"))
    ctx = int(os.environ.get("HESSIAN_CTX_SIZE", "4096"))
    batch = int(os.environ.get("HESSIAN_BATCH_SIZE", "1"))
    # Upstream defaults large_batch_size=512, which materializes ~17 GB of BF16
    # embeddings for Llama-2-7B at ctx=4096. Use 64 by default on Colab; the
    # upstream script already accumulates Hessians correctly across splits.
    large_batch = int(os.environ.get("HESSIAN_LARGE_BATCH_SIZE", "64"))
    sample_proc = int(os.environ.get("HESSIAN_SAMPLE_PROC", "4"))
    if large_batch > devset:
        large_batch = devset

    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # Runtime collector imports `scripts.qtip_hessian_utils`, so expose repo root.
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    print(f"[HESSIAN] PYTHONPATH={env['PYTHONPATH']}")
    print(f"[HESSIAN] devset={devset} ctx={ctx} batch={batch} large_batch={large_batch} sample_proc={sample_proc}")

    # torchrun initializes the single-GPU NCCL process group expected by upstream.
    cmd = [
        env_python(), "-m", "torch.distributed.run", "--nproc_per_node=1",
        str(runtime_script),
        "--base_model", MODEL_ID,
        "--save_path", str(out),
        "--batch_size", str(batch),
        "--large_batch_size", str(large_batch),
        "--devset_size", str(devset),
        "--ctx_size", str(ctx),
        "--sample_proc", str(sample_proc),
    ]
    run(cmd, cwd=ROOT, env=env)
    validate_hessian_dir(out)
    print(f"[OK] generated upstream-format Hessians: {out}")
    return out


if __name__ == "__main__":
    print(main())

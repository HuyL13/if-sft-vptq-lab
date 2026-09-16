from __future__ import annotations

import os
from pathlib import Path

from .common import ARTIFACTS, MODEL_ID, ROOT, UPSTREAM, clean_env, run
from .quantize_vptq import validate_hessian_dir
from .setup_env import env_python


def build_runtime_collector(quip: Path) -> Path:
    """Create a runtime copy of QuIP#'s official offline Hessian collector.

    VPTQ's own hessian loader says its format comes from QuIP#. The QuIP# collector
    writes the exact fields VPTQ consumes: flatH, mu, n, ct. We keep the upstream
    collector body unchanged and replace only `from lib import utils` with a tiny
    source-pinned shim containing the four utility functions this collector uses.
    This avoids importing unrelated QuIP# inference/codebook CUDA extensions.
    """
    upstream_script = quip / "quantize_llama" / "hessian_offline_llama.py"
    if not upstream_script.exists():
        raise RuntimeError("Pinned QuIP# Hessian collector missing")

    text = upstream_script.read_text(encoding="utf-8")
    old = "from lib import utils"
    new = "from scripts import quip_hessian_utils as utils"
    if old not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; expected import not found")

    runtime_dir = ARTIFACTS / "runtime_patches"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_script = runtime_dir / "hessian_offline_llama.py"
    runtime_script.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[HESSIAN] runtime collector: {runtime_script}")
    print("[HESSIAN] source: pinned Cornell-RelaxML/quip-sharp hessian_offline_llama.py")
    print("[HESSIAN] patch: only `from lib import utils` -> source-pinned minimal Hessian shim")
    return runtime_script


def main() -> Path:
    explicit = os.environ.get("VPTQ_HESSIAN_DIR")
    if explicit:
        path = Path(explicit).resolve()
        validate_hessian_dir(path, inspect_contents=True)
        print(f"[HESSIAN] using provided directory: {path}")
        return path

    out = ARTIFACTS / "hessians_llama2_7b_fingerprinted"
    try:
        validate_hessian_dir(out, inspect_contents=True)
        print(f"[SKIP] Hessians already complete: {out}")
        return out
    except RuntimeError:
        pass

    out.mkdir(parents=True, exist_ok=True)
    quip = UPSTREAM / "quip-sharp"
    runtime_script = build_runtime_collector(quip)

    # QuIP# upstream defaults: devset=256, ctx=4096, batch=2, chunk=256.
    # A Llama-2-7B activation cache at devset=256 is ~8.6 GB BF16 before model/RAM
    # overhead. Use 64 samples by default for a Colab-safe attack-screening run while
    # preserving the same collector/math. Set HESSIAN_DEVSET_SIZE=256 for the exact
    # upstream sample count.
    devset = int(os.environ.get("HESSIAN_DEVSET_SIZE", "64"))
    ctx = int(os.environ.get("HESSIAN_CTX_SIZE", "4096"))
    batch = int(os.environ.get("HESSIAN_BATCH_SIZE", "1"))
    chunk = int(os.environ.get("HESSIAN_CHUNK_SIZE", str(devset)))
    sample_proc = int(os.environ.get("HESSIAN_SAMPLE_PROC", "4"))

    if devset < 1 or ctx < 1 or batch < 1 or chunk < 1:
        raise ValueError("Hessian sizes must be positive")
    chunk = min(chunk, devset)
    if devset % batch != 0 or chunk % batch != 0:
        raise ValueError("HESSIAN_DEVSET_SIZE and HESSIAN_CHUNK_SIZE must be divisible by HESSIAN_BATCH_SIZE")

    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    print(f"[HESSIAN] PYTHONPATH={env['PYTHONPATH']}")
    print(f"[HESSIAN] devset={devset} ctx={ctx} batch={batch} chunk={chunk} sample_proc={sample_proc}")

    cmd = [
        env_python(), str(runtime_script),
        "--base_model", MODEL_ID,
        "--save_path", str(out),
        "--batch_size", str(batch),
        "--devset_size", str(devset),
        "--ctx_size", str(ctx),
        "--chunk_size", str(chunk),
        "--sample_proc", str(sample_proc),
    ]
    # QuIP# collector manages its own multiprocessing/GPU workers; do not wrap it in
    # torchrun. With one visible Colab GPU, ngpus resolves to 1 inside upstream code.
    run(cmd, cwd=ROOT, env=env)
    validate_hessian_dir(out, inspect_contents=True)
    print(f"[OK] generated QuIP#-format Hessians accepted by VPTQ: {out}")
    return out


if __name__ == "__main__":
    print(main())

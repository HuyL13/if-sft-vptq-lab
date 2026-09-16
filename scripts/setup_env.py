from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .common import (
    ROOT,
    UPSTREAM,
    ENV,
    VPTQ_URL,
    VPTQ_REF,
    MF_URL,
    MF_REF,
    QTIP_URL,
    QTIP_REF,
    clone_pinned,
    clean_env,
    ensure_dirs,
    output,
    run,
    write_json,
)


def system_snapshot() -> dict:
    code = (
        "import json,sys; d={'python':sys.executable,'version':sys.version};\n"
        "try:\n import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda,torch_file=torch.__file__)\n"
        "except Exception as e: d['torch_error']=repr(e)\n"
        "print(json.dumps(d))"
    )
    return json.loads(output([sys.executable, "-c", code]))


def env_python() -> Path:
    # Kept as a compatibility helper for the rest of the pipeline.
    # There is intentionally NO venv: every stage uses the current Colab Python.
    return Path(sys.executable)


def _torch_identity(py: Path | str) -> list[str]:
    return output([
        py,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()


def install_dependencies() -> None:
    py = env_python()
    env = clean_env()
    before = _torch_identity(py)
    print(f"[GUARD] using Colab system Python: {py}")
    print(f"[GUARD] using existing Torch {before[0]} / CUDA {before[1]}")
    print(f"[GUARD] Torch path: {before[2]}")
    print("[GUARD] this setup never runs pip install torch/torchvision/torchaudio")

    # Install only non-Torch dependencies into the current Colab environment.
    # Packages with their own Torch dependency are installed with --no-deps where
    # necessary so pip cannot replace the preinstalled Colab Torch wheel.
    run([
        py,
        "-m",
        "pip",
        "install",
        "accelerate",
        "transformers>=4.45,<5",
        "datasets",
        "scipy",
        "numpy",
        "pyyaml",
        "huggingface-hub",
        "tqdm",
        "ninja",
        "fschat",
    ], env=env)

    # sentence-transformers depends on torch, but the experiment only needs the
    # package import required by upstream run_vptq.py. Prevent dependency resolution
    # from touching Torch.
    run([
        py,
        "-m",
        "pip",
        "install",
        "sentence_transformers",
        "--no-deps",
    ], env=env)

    run([
        py,
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        "https://pypi.nvidia.com",
        "cuml-cu12==24.12.*",
        "--no-deps",
    ], env=env)

    # Upstream VPTQ get_llama() explicitly requests flash_attention_2.
    # Build against the already-loaded Colab Torch; never let pip install another one.
    run([
        py,
        "-m",
        "pip",
        "install",
        "flash-attn==2.5.8",
        "--no-build-isolation",
        "--no-deps",
    ], env=env)

    # Use Microsoft's Python algorithm path without compiling the optional VPTQ CUDA
    # extension. --no-deps prevents its package metadata from altering Torch.
    env2 = dict(env)
    env2["SKIP_COMPILE"] = "1"
    run([
        py,
        "-m",
        "pip",
        "install",
        "-e",
        str(UPSTREAM / "VPTQ"),
        "--no-build-isolation",
        "--no-deps",
    ], env=env2)

    after = _torch_identity(py)
    if after != before:
        raise RuntimeError(
            "Torch changed during setup, which is forbidden. "
            f"before={before!r}, after={after!r}"
        )
    print("[GUARD] Torch unchanged after dependency installation")


def smoke() -> None:
    py = env_python()
    code = (
        "import torch,vptq,transformers,flash_attn; "
        "print('python',__import__('sys').executable); "
        "print('torch',torch.__version__,'cuda',torch.version.cuda,'torch_file',torch.__file__); "
        "print('gpu',torch.cuda.get_device_name(0)); "
        "print('vptq',vptq.__file__); "
        "print('transformers',transformers.__version__); "
        "print('flash_attn',flash_attn.__version__)"
    )
    run([py, "-c", code], env=clean_env())


def main() -> None:
    ensure_dirs()
    before = system_snapshot()
    if "torch_error" in before:
        raise RuntimeError(f"Colab system Torch is required but unavailable: {before['torch_error']}")
    write_json(ENV / "system-before.json", before)

    clone_pinned(VPTQ_URL, UPSTREAM / "VPTQ", VPTQ_REF)
    clone_pinned(MF_URL, UPSTREAM / "Model-Fingerprint", MF_REF)
    clone_pinned(QTIP_URL, UPSTREAM / "qtip", QTIP_REF)

    install_dependencies()
    smoke()

    after = system_snapshot()
    write_json(ENV / "system-after.json", after)
    if before.get("torch") != after.get("torch") or before.get("cuda") != after.get("cuda") or before.get("torch_file") != after.get("torch_file"):
        raise RuntimeError("Colab Torch/CUDA changed during setup")
    print("[OK] setup complete: no venv created, Colab system Torch reused unchanged")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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

ENV_PREFIX = ROOT / ".venv-vptq"


def system_snapshot() -> dict:
    code = (
        "import json,sys; d={'python':sys.executable,'version':sys.version};\n"
        "try:\n import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda,torch_file=torch.__file__)\n"
        "except Exception as e: d['torch_error']=repr(e)\n"
        "print(json.dumps(d))"
    )
    return json.loads(output([sys.executable, "-c", code]))


def env_python() -> Path:
    return ENV_PREFIX / "bin" / "python"


def create_env() -> None:
    # IMPORTANT: do not install/reinstall/downgrade Torch.  The experiment must reuse
    # the Torch/CUDA stack already provided by Colab.  A venv with
    # --system-site-packages gives the experiment access to that exact Torch build
    # while keeping the remaining Python dependencies local to this repository.
    if not env_python().exists():
        run([
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(ENV_PREFIX),
        ])

    py = env_python()
    env = clean_env()

    # Colab images occasionally create a venv without a working pip/ensurepip.
    # Bootstrap pip into the venv without touching Torch or the system interpreter.
    probe = run([py, "-m", "pip", "--version"], env=env, check=False)
    if probe.returncode != 0:
        system_pip = shutil.which("pip") or shutil.which("pip3")
        if not system_pip:
            raise RuntimeError("venv has no pip and no system pip was found")
        run([system_pip, "--python", str(py), "install", "--upgrade", "pip", "setuptools", "wheel"], env=env)
    else:
        run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "ninja"], env=env)

    # Guard before installing anything else: the venv must see the exact same Torch
    # package as Colab's system Python.
    system_torch = output([
        sys.executable,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()
    env_torch = output([
        py,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()
    if system_torch != env_torch:
        raise RuntimeError(
            "The experiment venv is not reusing Colab system Torch exactly. "
            f"system={system_torch!r}, venv={env_torch!r}"
        )
    print(f"[GUARD] reusing Colab Torch {system_torch[0]} / CUDA {system_torch[1]}")

    # Install only non-Torch dependencies.  --no-deps on packages that may declare
    # their own Torch constraints prevents pip from replacing the Colab Torch wheel.
    run([
        py,
        "-m",
        "pip",
        "install",
        "accelerate",
        "transformers>=4.45,<5",
        "datasets",
        "sentence_transformers",
        "fschat",
        "scipy",
        "numpy",
        "pyyaml",
        "huggingface-hub",
        "tqdm",
        "ninja",
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

    # Upstream get_llama() requests flash_attention_2.  Build it against the existing
    # Colab Torch; never allow pip to pull a different Torch as a dependency.
    run([
        py,
        "-m",
        "pip",
        "install",
        "flash-attn==2.5.8",
        "--no-build-isolation",
        "--no-deps",
    ], env=env)

    # Algorithm path has a pure-PyTorch dequant fallback, so the optional VPTQ CUDA
    # extension is unnecessary for correctness.  Keep Microsoft's Python code path
    # intact while avoiding an extra CUDA build during setup.
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

    # Re-check after dependency installation so a transitive pip dependency cannot
    # silently replace Torch.
    env_torch_after = output([
        py,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()
    if env_torch_after != system_torch:
        raise RuntimeError(
            "Torch changed during setup, which is forbidden. "
            f"before={system_torch!r}, after={env_torch_after!r}"
        )


def smoke() -> None:
    py = env_python()
    code = (
        "import torch,vptq,transformers,flash_attn; "
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
    # VPTQ's Hessian loader explicitly derives from Cornell's QuIP#/QTIP format.
    # QTIP is pinned only to generate compatible Hessians for the fingerprinted model.
    clone_pinned(QTIP_URL, UPSTREAM / "qtip", QTIP_REF)

    create_env()
    smoke()

    after = system_snapshot()
    write_json(ENV / "system-after.json", after)
    if before != after:
        raise RuntimeError("System Python/Torch snapshot changed during setup")
    print("[OK] VPTQ environment ready; reused Colab system Torch without reinstalling it")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from .common import (
    ENV,
    HYPERQUANT_REF,
    HYPERQUANT_URL,
    MF_REF,
    MF_URL,
    ROOT,
    UPSTREAM,
    clean_env,
    clone_pinned,
    ensure_dirs,
    output,
    run,
    write_json,
)


def env_python() -> Path:
    return Path(sys.executable)


def _torch_identity() -> list[str]:
    return output([
        sys.executable,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()


def _assert_torch_unchanged(expected: list[str], stage: str) -> None:
    got = _torch_identity()
    if got != expected:
        raise RuntimeError(f"Torch changed during {stage}: expected={expected!r}, got={got!r}")


def _gpu_snapshot() -> dict:
    code = r'''
import json, shutil, subprocess, torch
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
print(json.dumps({
  'python': __import__('sys').executable,
  'torch': torch.__version__,
  'torch_cuda': torch.version.cuda,
  'torch_file': torch.__file__,
  'cuda_available': torch.cuda.is_available(),
  'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'compute_capability': list(cc) if cc else None,
  'nvcc': shutil.which('nvcc'),
}))
'''
    return json.loads(output([sys.executable, "-c", code]))


def patch_hyperquant_for_runtime_gpu() -> None:
    """Patch only the upstream compile arch so HyperQuant builds on A100/SM80.

    Upstream currently hard-codes `-arch=sm_90a` in both llama integration
    extensions even though its README states SM80+ support. The IF-SFT experiment
    uses the upstream INT8 path; we make its arch flag derive from the active GPU.

    The target file is reset to the pinned commit before every patch, making this
    setup idempotent and preventing the double-patching failure seen in the older
    VPTQ experiment.
    """
    repo = UPSTREAM / "HyperQuant"
    target = repo / "integrations" / "llama" / "int8_linear.py"
    run(["git", "checkout", HYPERQUANT_REF, "--", str(target.relative_to(repo))], cwd=repo)

    text = target.read_text(encoding="utf-8")
    marker = '_RHT_SIZES       = (2048, 1024, 512, 256)   # checked in order, largest first\n'
    arch_line = '_CUDA_ARCH = f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"\n'
    if marker not in text:
        raise RuntimeError("Pinned HyperQuant int8_linear.py changed unexpectedly (RHT marker missing)")
    text = text.replace(marker, marker + arch_line, 1)

    old = 'extra_cuda_cflags=["-O3", "-arch=sm_90a"],'
    new = 'extra_cuda_cflags=["-O3", f"-arch={_CUDA_ARCH}"],'
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one HyperQuant INT8 hard-coded sm_90a flag, found {text.count(old)}")
    text = text.replace(old, new, 1)
    target.write_text(text, encoding="utf-8")

    subprocess.run([sys.executable, "-m", "py_compile", str(target)], check=True)
    print(f"[PATCH] HyperQuant INT8 CUDA arch is runtime-selected via {_gpu_snapshot()['compute_capability']}")
    print("[PATCH] patched HyperQuant int8_linear.py compiles: OK")


def install_dependencies() -> None:
    torch_before = _torch_identity()
    env = clean_env()
    print(f"[GUARD] system Python: {sys.executable}")
    print(f"[GUARD] existing Torch: {torch_before[0]} / CUDA {torch_before[1]}")
    print(f"[GUARD] Torch path: {torch_before[2]}")
    print("[GUARD] HyperQuant setup never installs/reinstalls torch, torchvision, or torchaudio")

    # Keep the dependency surface intentionally small. HyperQuant itself is installed
    # editable with --no-deps so its `torch>=2.1` requirement cannot cause pip to
    # replace the Colab-provided Torch build.
    run([
        sys.executable, "-m", "pip", "install",
        "transformers>=4.45,<5",
        "datasets",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "huggingface-hub",
        "shortuuid",
        "markdown2",
    ], env=env)
    _assert_torch_unchanged(torch_before, "core dependency installation")

    run([sys.executable, "-m", "pip", "install", "fschat==0.2.36", "--no-deps"], env=env)
    _assert_torch_unchanged(torch_before, "FastChat installation")

    run([
        sys.executable, "-m", "pip", "install", "-e", str(UPSTREAM / "HyperQuant"),
        "--no-build-isolation", "--no-deps",
    ], env=env)
    _assert_torch_unchanged(torch_before, "HyperQuant installation")
    print("[GUARD] Torch unchanged after HyperQuant setup")


def main() -> None:
    ensure_dirs()
    before = _gpu_snapshot()
    write_json(ENV / "hyperquant-system-before.json", before)
    if not before["cuda_available"]:
        raise RuntimeError("CUDA GPU is required")
    cc = tuple(before["compute_capability"])
    if cc < (8, 0):
        raise RuntimeError(f"HyperQuant upstream requires SM80+; detected compute capability {cc}")
    if before["nvcc"] is None:
        raise RuntimeError("nvcc is required because HyperQuant builds its CUDA extension on first use")

    clone_pinned(HYPERQUANT_URL, UPSTREAM / "HyperQuant", HYPERQUANT_REF)
    clone_pinned(MF_URL, UPSTREAM / "Model-Fingerprint", MF_REF)
    patch_hyperquant_for_runtime_gpu()
    install_dependencies()

    after = _gpu_snapshot()
    write_json(ENV / "hyperquant-system-after.json", after)
    for key in ("torch", "torch_cuda", "torch_file"):
        if before[key] != after[key]:
            raise RuntimeError(f"Colab Torch changed during setup ({key}: {before[key]} -> {after[key]})")
    print("[OK] HyperQuant setup complete: pinned upstream, no venv, no Torch reinstall")


if __name__ == "__main__":
    main()

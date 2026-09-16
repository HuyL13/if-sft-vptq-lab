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
    # Intentionally no venv: every stage uses the current Colab Python/Torch stack.
    return Path(sys.executable)


def _torch_identity(py: Path | str) -> list[str]:
    return output([
        py,
        "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.__file__)",
    ]).splitlines()


def _assert_torch_unchanged(py: Path | str, expected: list[str], stage: str) -> None:
    got = _torch_identity(py)
    if got != expected:
        raise RuntimeError(
            f"Torch changed during {stage}, which is forbidden. expected={expected!r}, got={got!r}"
        )


def patch_vptq_for_colab() -> None:
    """Apply the smallest compatibility patch needed for current Colab.

    Microsoft's pinned algorithm hard-codes flash_attention_2 when loading LLaMA.
    flash-attn==2.5.8 is from the upstream 2024 environment and is not a sensible
    dependency to build against current Colab Python 3.13 / Torch 2.11.  Transformers'
    SDPA backend is sufficient for loading/quantization and does not alter VPTQ math.
    """
    llama_py = UPSTREAM / "VPTQ" / "vptq" / "models" / "llama.py"
    text = llama_py.read_text(encoding="utf-8")
    old = 'attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16'
    new = 'attn_implementation="sdpa", torch_dtype=torch.bfloat16'
    if old in text:
        llama_py.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[PATCH] {llama_py}: flash_attention_2 -> sdpa (Colab compatibility)")
    elif new in text:
        print(f"[PATCH] already applied: {llama_py}")
    else:
        raise RuntimeError("Pinned VPTQ llama.py changed unexpectedly; refusing an unverified patch")


def install_dependencies() -> None:
    py = env_python()
    env = clean_env()
    torch_before = _torch_identity(py)
    print(f"[GUARD] using Colab system Python: {py}")
    print(f"[GUARD] using existing Torch {torch_before[0]} / CUDA {torch_before[1]}")
    print(f"[GUARD] Torch path: {torch_before[2]}")
    print("[GUARD] this setup never runs pip install torch/torchvision/torchaudio")

    # Core dependencies. FastChat is deliberately installed separately with --no-deps
    # because its markdown2[all] extra pulls legacy wavedrom, which fails on Python 3.13.
    run([
        py, "-m", "pip", "install",
        "accelerate",
        "transformers>=4.45,<5",
        "datasets",
        "scipy",
        "numpy",
        "pyyaml",
        "huggingface-hub",
        "tqdm",
        "ninja",
        "shortuuid",
        "markdown2",
    ], env=env)
    _assert_torch_unchanged(py, torch_before, "core dependency installation")

    run([py, "-m", "pip", "install", "fschat==0.2.36", "--no-deps"], env=env)
    run([py, "-m", "pip", "install", "sentence_transformers", "--no-deps"], env=env)
    _assert_torch_unchanged(py, torch_before, "FastChat/SentenceTransformers installation")

    # IMPORTANT: the VPTQ paper code used cuML 23.12/24.12, but those releases do
    # not provide a CPython 3.13 wheel. On current Colab, asking for 24.12 downloads
    # NVIDIA's tiny source/redirect package and fails during metadata generation.
    # RAPIDS 26.8 ships a CPython 3.11+ abi3 wheel and supports CUDA 12.x.
    # Force binary wheels so setup can never silently fall back to a source stub.
    run([
        py, "-m", "pip", "install",
        "cuml-cu12==26.8.0",
        "--only-binary=:all:",
    ], env=env)
    _assert_torch_unchanged(py, torch_before, "RAPIDS cuML installation")

    # Install Microsoft's VPTQ package itself, but do not compile the optional custom
    # inference extension. The algorithm has a pure-PyTorch dequantization fallback.
    env2 = dict(env)
    env2["SKIP_COMPILE"] = "1"
    run([
        py, "-m", "pip", "install", "-e", str(UPSTREAM / "VPTQ"),
        "--no-build-isolation", "--no-deps",
    ], env=env2)
    _assert_torch_unchanged(py, torch_before, "VPTQ installation")

    print("[GUARD] Torch unchanged after all dependency installation")


def smoke() -> None:
    py = env_python()
    code = (
        "import sys,torch,vptq,transformers,cuml,cupy; "
        "from fastchat.model.model_adapter import get_conversation_template; "
        "assert get_conversation_template('vicuna') is not None; "
        "from cuml.cluster import KMeans; "
        "print('python',sys.executable); "
        "print('torch',torch.__version__,'cuda',torch.version.cuda,'torch_file',torch.__file__); "
        "print('gpu',torch.cuda.get_device_name(0)); "
        "print('vptq',vptq.__file__); "
        "print('transformers',transformers.__version__); "
        "print('cuml',cuml.__version__); "
        "print('cupy',cupy.__version__); "
        "print('fastchat vicuna template: OK'); "
        "print('cuml KMeans import: OK')"
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

    patch_vptq_for_colab()
    install_dependencies()
    smoke()

    after = system_snapshot()
    write_json(ENV / "system-after.json", after)
    if before.get("torch") != after.get("torch") or before.get("cuda") != after.get("cuda") or before.get("torch_file") != after.get("torch_file"):
        raise RuntimeError("Colab Torch/CUDA changed during setup")
    print("[OK] setup complete: no venv, no Torch reinstall, Python 3.13-compatible cuML")


if __name__ == "__main__":
    main()

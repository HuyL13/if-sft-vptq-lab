from __future__ import annotations

import os
from pathlib import Path
import sys

from .common import ROOT, UPSTREAM, ENV, VPTQ_URL, VPTQ_REF, MF_URL, MF_REF, clone_pinned, clean_env, ensure_dirs, output, run, write_json

MAMBA = ROOT / ".tools" / "bin" / "micromamba"
ENV_PREFIX = ROOT / ".venv-vptq"


def system_snapshot() -> dict:
    code = "import json,sys; d={'python':sys.executable,'version':sys.version};\ntry:\n import torch; d.update(torch=torch.__version__,cuda=torch.version.cuda)\nexcept Exception as e: d['torch_error']=repr(e)\nprint(json.dumps(d))"
    import json
    return json.loads(output([sys.executable, "-c", code]))


def ensure_micromamba() -> None:
    if MAMBA.exists():
        return
    MAMBA.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {MAMBA.parent} --strip-components=1 bin/micromamba"
    run(["bash", "-lc", cmd])
    MAMBA.chmod(0o755)


def env_python() -> Path:
    return ENV_PREFIX / "bin" / "python"


def create_env() -> None:
    ensure_micromamba()
    if not env_python().exists():
        run([MAMBA, "create", "-y", "-p", ENV_PREFIX, "python=3.10", "pip", "ninja", "-c", "conda-forge"])
    py = env_python()
    env = clean_env()
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "ninja"], env=env)
    # Match Microsoft's released algorithm environment in an isolated prefix.
    run([py, "-m", "pip", "install", "torch==2.5.1", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu121"], env=env)
    run([py, "-m", "pip", "install", "accelerate", "transformers>=4.45,<5", "datasets", "sentence_transformers", "fschat", "scipy", "numpy", "pyyaml", "huggingface-hub"], env=env)
    run([py, "-m", "pip", "install", "--extra-index-url", "https://pypi.nvidia.com", "cuml-cu12==24.12.*"], env=env)
    # get_llama() in the upstream algorithm explicitly requests flash_attention_2.
    # Build only this required dependency, not the optional VPTQ CUDA inference extension.
    run([py, "-m", "pip", "install", "flash-attn==2.5.8", "--no-build-isolation"], env=env)
    env2 = dict(env)
    env2["SKIP_COMPILE"] = "1"
    run([py, "-m", "pip", "install", "-e", str(UPSTREAM / "VPTQ"), "--no-build-isolation"], env=env2)
    run([py, "-m", "pip", "check"], env=env)


def smoke() -> None:
    py = env_python()
    code = "import torch,vptq,transformers,flash_attn; print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.get_device_name(0)); print('vptq',vptq.__file__); print('transformers',transformers.__version__); print('flash_attn',flash_attn.__version__)"
    run([py, "-c", code], env=clean_env())


def main() -> None:
    ensure_dirs()
    before = system_snapshot()
    write_json(ENV / "system-before.json", before)
    clone_pinned(VPTQ_URL, UPSTREAM / "VPTQ", VPTQ_REF)
    clone_pinned(MF_URL, UPSTREAM / "Model-Fingerprint", MF_REF)
    create_env()
    smoke()
    after = system_snapshot()
    write_json(ENV / "system-after.json", after)
    if before != after:
        raise RuntimeError("System Python/Torch snapshot changed during isolated setup")
    print("[OK] isolated VPTQ environment ready; system Torch unchanged")


if __name__ == "__main__":
    main()

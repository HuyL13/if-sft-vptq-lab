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
    QUIP_URL,
    QUIP_REF,
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


def _restore_vptq_patch_targets() -> None:
    """Reset runtime-patched VPTQ files to the pinned upstream commit first."""
    repo = UPSTREAM / "VPTQ"
    targets = [
        "vptq/models/llama.py",
        "vptq/quantizer.py",
        "vptq/vptq.py",
    ]
    run(["git", "checkout", VPTQ_REF, "--", *targets], cwd=repo)
    print("[PATCH] restored VPTQ patch targets to pinned upstream before applying Colab compatibility patches")


def patch_vptq_for_colab() -> None:
    _restore_vptq_patch_targets()

    # 1) Avoid the legacy flash-attn build in the pinned environment.
    llama_py = UPSTREAM / "VPTQ" / "vptq" / "models" / "llama.py"
    text = llama_py.read_text(encoding="utf-8")
    old = 'attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16'
    new = 'attn_implementation="sdpa", torch_dtype=torch.bfloat16'
    if old not in text:
        raise RuntimeError("Pinned VPTQ llama.py changed unexpectedly; refusing an unverified patch")
    llama_py.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[PATCH] {llama_py}: flash_attention_2 -> sdpa (Colab compatibility)")

    # 2) cuML 26.8 requires sample_weight to be a 1-D NumPy array.
    # Upstream main-codebook KMeans already reduces reshaped Hessian weights with
    # mean(dim=1); the residual-codebook path forgot that reduction and passes a
    # [num_vectors, vector_len] tensor. Preserve the intended weighting semantics by
    # applying the same mean(dim=1) reduction in the residual path, then convert the
    # resulting 1-D tensor to float32 NumPy in both paths.
    quantizer_py = UPSTREAM / "VPTQ" / "vptq" / "quantizer.py"
    qtext = quantizer_py.read_text(encoding="utf-8")

    main_old = (
        "vector_weights = vector_weights.mean(dim=1) if vector_weights is not None else None\n"
        "                # convert to numpy and float32 to avoid error\n"
        "                sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()\n"
        "                with cupy.cuda.Device(vector_weights.device.index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    main_new = (
        "vector_weights = vector_weights.mean(dim=1) if vector_weights is not None else None\n"
        "                # cuML 26.8 requires 1-D NumPy sample weights.\n"
        "                device_index = vector_weights.device.index if (vector_weights is not None and vector_weights.is_cuda) else 0\n"
        "                if vector_weights is not None:\n"
        "                    vector_weights = vector_weights.to(torch.float32).detach().cpu().numpy()\n"
        "                sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()\n"
        "                with cupy.cuda.Device(device_index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    if qtext.count(main_old) != 1:
        raise RuntimeError("Unexpected pinned VPTQ main-codebook KMeans pattern")
    qtext = qtext.replace(main_old, main_new, 1)

    residual_old = (
        "sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()\n"
        "                with cupy.cuda.Device(vector_weights.device.index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    residual_new = (
        "vector_weights = vector_weights.mean(dim=1) if vector_weights is not None else None\n"
        "                device_index = vector_weights.device.index if (vector_weights is not None and vector_weights.is_cuda) else 0\n"
        "                if vector_weights is not None:\n"
        "                    vector_weights = vector_weights.to(torch.float32).detach().cpu().numpy()\n"
        "                sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()\n"
        "                with cupy.cuda.Device(device_index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    if qtext.count(residual_old) != 1:
        raise RuntimeError("Unexpected pinned VPTQ residual-codebook KMeans pattern")
    qtext = qtext.replace(residual_old, residual_new, 1)

    quantizer_py.write_text(qtext, encoding="utf-8")
    print(f"[PATCH] {quantizer_py}: cuML 26.8 1-D NumPy sample weights fixed for main + residual KMeans")

    # 3) --inv_hessian_path is optional in layer_quantizer, and VPTQ.vptq() already
    # knows how to derive the inverse from the ordinary Hessian when passed None.
    vptq_py = UPSTREAM / "VPTQ" / "vptq" / "vptq.py"
    vtext = vptq_py.read_text(encoding="utf-8")

    initial_old = "inv_hessian = self.inv_hessian.clone().to('cpu')"
    initial_new = "inv_hessian = self.inv_hessian.clone().to('cpu') if self.inv_hessian is not None else None"
    if vtext.count(initial_old) != 1:
        raise RuntimeError("Unexpected pinned VPTQ initial inverse-Hessian clone pattern")
    vtext = vtext.replace(initial_old, initial_new, 1)

    cpu_old = "        inv_hessian = inv_hessian.to('cpu')\n        # end of weight and hessian preprocess"
    cpu_new = "        if inv_hessian is not None:\n            inv_hessian = inv_hessian.to('cpu')\n        # end of weight and hessian preprocess"
    if vtext.count(cpu_old) != 1:
        raise RuntimeError("Unexpected pinned VPTQ inverse-Hessian CPU-transfer pattern")
    vtext = vtext.replace(cpu_old, cpu_new, 1)

    round_old = "_inv_hessian = inv_hessian.clone().to(self.dev)"
    round_new = "_inv_hessian = inv_hessian.clone().to(self.dev) if inv_hessian is not None else None"
    round_count = vtext.count(round_old)
    if round_count != 2:
        raise RuntimeError(f"Expected 2 VPTQ round inverse-Hessian clones, found {round_count}")
    vtext = vtext.replace(round_old, round_new)

    vptq_py.write_text(vtext, encoding="utf-8")
    print(f"[PATCH] {vptq_py}: optional inverse Hessian fixed at initial + {round_count} round call sites")

    run([sys.executable, "-m", "py_compile", str(llama_py), str(quantizer_py), str(vptq_py)])
    print("[PATCH] patched VPTQ files compile: OK")


def install_dependencies() -> None:
    py = env_python()
    env = clean_env()
    torch_before = _torch_identity(py)
    print(f"[GUARD] using Colab system Python: {py}")
    print(f"[GUARD] using existing Torch {torch_before[0]} / CUDA {torch_before[1]}")
    print(f"[GUARD] Torch path: {torch_before[2]}")
    print("[GUARD] this setup never runs pip install torch/torchvision/torchaudio")

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
        "glog",
    ], env=env)
    _assert_torch_unchanged(py, torch_before, "core dependency installation")

    run([py, "-m", "pip", "install", "fschat==0.2.36", "--no-deps"], env=env)
    run([py, "-m", "pip", "install", "sentence_transformers", "--no-deps"], env=env)
    _assert_torch_unchanged(py, torch_before, "FastChat/SentenceTransformers installation")

    run([
        py, "-m", "pip", "install",
        "cuml-cu12==26.8.0",
        "--only-binary=:all:",
    ], env=env)
    _assert_torch_unchanged(py, torch_before, "RAPIDS cuML installation")

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
        "import sys,torch,vptq,transformers,cuml,cupy,glog; "
        "from fastchat.model.model_adapter import get_conversation_template; "
        "assert get_conversation_template('vicuna') is not None; "
        "from cuml.cluster import KMeans; "
        "from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask; "
        "print('python',sys.executable); "
        "print('torch',torch.__version__,'cuda',torch.version.cuda,'torch_file',torch.__file__); "
        "print('gpu',torch.cuda.get_device_name(0)); "
        "print('vptq',vptq.__file__); "
        "print('transformers',transformers.__version__); "
        "print('cuml',cuml.__version__); "
        "print('cupy',cupy.__version__); "
        "print('fastchat vicuna template: OK'); "
        "print('cuml KMeans import: OK'); "
        "print('transformers causal-mask helper: OK')"
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
    clone_pinned(QUIP_URL, UPSTREAM / "quip-sharp", QUIP_REF)

    patch_vptq_for_colab()
    install_dependencies()
    smoke()

    after = system_snapshot()
    write_json(ENV / "system-after.json", after)
    if before.get("torch") != after.get("torch") or before.get("cuda") != after.get("cuda") or before.get("torch_file") != after.get("torch_file"):
        raise RuntimeError("Colab Torch/CUDA changed during setup")
    print("[OK] setup complete: no venv, no Torch reinstall, Python 3.13-compatible VPTQ/cuML")


if __name__ == "__main__":
    main()

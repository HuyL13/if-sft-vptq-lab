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


def patch_vptq_for_colab() -> None:
    # 1) Current Colab does not need the legacy flash-attn==2.5.8 build used by the
    # pinned 2024 environment. This changes only the attention implementation used
    # while loading/running LLaMA; it does not change VPTQ quantization math.
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

    # 2) The pinned VPTQ code passes a CUDA torch.Tensor as cuML KMeans
    # `sample_weight`. Current Colab-compatible cuML 26.8 rejects that dtype.
    # Convert ONLY sample_weight to float32 NumPy before KMeans.fit. The values and
    # weighting are unchanged; sub_vectors are already converted to NumPy upstream.
    quantizer_py = UPSTREAM / "VPTQ" / "vptq" / "quantizer.py"
    qtext = quantizer_py.read_text(encoding="utf-8")
    old_fit = (
        "with cupy.cuda.Device(vector_weights.device.index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    new_fit = (
        "device_index = vector_weights.device.index if (vector_weights is not None and vector_weights.is_cuda) else 0\n"
        "                if vector_weights is not None:\n"
        "                    vector_weights = vector_weights.to(torch.float32).detach().cpu().numpy()\n"
        "                with cupy.cuda.Device(device_index):\n"
        "                    _kmeans.fit(sub_vectors, sample_weight=vector_weights)"
    )
    count = qtext.count(old_fit)
    if count:
        qtext = qtext.replace(old_fit, new_fit)
        quantizer_py.write_text(qtext, encoding="utf-8")
        print(f"[PATCH] {quantizer_py}: converted cuML sample_weight Torch->NumPy at {count} call site(s)")
    elif "vector_weights = vector_weights.to(torch.float32).detach().cpu().numpy()" in qtext:
        print(f"[PATCH] already applied: {quantizer_py}")
    else:
        raise RuntimeError("Pinned VPTQ quantizer.py changed unexpectedly; cuML compatibility patch not applied")

    # 3) Pinned VPTQ declares inverse Hessians optional: layer_quantizer passes
    # inv_hessian=None when --inv_hessian_path is omitted, and VPTQ.vptq() contains
    # the fallback that computes the inverse/Cholesky form from the ordinary Hessian.
    # But fast_vector_quant() unconditionally clones self.inv_hessian before reaching
    # that fallback, causing AttributeError: 'NoneType' object has no attribute 'clone'.
    # Preserve None here so the existing upstream fallback is actually used.
    vptq_py = UPSTREAM / "VPTQ" / "vptq" / "vptq.py"
    vtext = vptq_py.read_text(encoding="utf-8")
    old_inv = "inv_hessian = self.inv_hessian.clone().to('cpu')"
    new_inv = "inv_hessian = self.inv_hessian.clone().to('cpu') if self.inv_hessian is not None else None"
    old_cpu = "inv_hessian = inv_hessian.to('cpu')\n        # end of weight and hessian preprocess"
    new_cpu = "if inv_hessian is not None:\n            inv_hessian = inv_hessian.to('cpu')\n        # end of weight and hessian preprocess"
    changed = False
    if old_inv in vtext:
        vtext = vtext.replace(old_inv, new_inv, 1)
        changed = True
    if old_cpu in vtext:
        vtext = vtext.replace(old_cpu, new_cpu, 1)
        changed = True
    if changed:
        vptq_py.write_text(vtext, encoding="utf-8")
        print(f"[PATCH] {vptq_py}: allow inv_hessian=None and use upstream on-the-fly inverse fallback")
    elif new_inv in vtext and new_cpu in vtext:
        print(f"[PATCH] already applied: {vptq_py}")
    else:
        raise RuntimeError("Pinned VPTQ vptq.py changed unexpectedly; inverse-Hessian compatibility patch not applied")


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

from __future__ import annotations

import os
from pathlib import Path

from .common import ARTIFACTS, MODEL_ID, ROOT, UPSTREAM, clean_env, run
from .quantize_vptq import validate_hessian_dir
from .setup_env import env_python


def build_runtime_collector(quip: Path) -> Path:
    """Create a runtime copy of QuIP#'s official offline Hessian collector.

    VPTQ consumes the QuIP# Hessian format (flatH, mu, n, ct). The pinned QuIP#
    collector predates the current Transformers LLaMA API, so we apply two narrow
    runtime-only compatibility patches while leaving the Hessian math unchanged:
      1) replace the heavyweight `lib.utils` import with our source-pinned shim;
      2) explicitly compute/pass rotary `position_embeddings`, required by modern
         Transformers when calling a LlamaDecoderLayer directly.
    """
    upstream_script = quip / "quantize_llama" / "hessian_offline_llama.py"
    if not upstream_script.exists():
        raise RuntimeError("Pinned QuIP# Hessian collector missing")

    text = upstream_script.read_text(encoding="utf-8")

    old_import = "from lib import utils"
    new_import = "from scripts import quip_hessian_utils as utils"
    if old_import not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; utils import not found")
    text = text.replace(old_import, new_import, 1)

    # Transformers 4.57+ no longer derives RoPE inside LlamaAttention when a decoder
    # layer is invoked directly. QuIP# calls each decoder layer directly, therefore
    # position_embeddings would otherwise be None and LlamaAttention crashes at
    # `cos, sin = position_embeddings`.
    old_sig = (
        "def forward_layer(layer, position_ids, attention_mask, bs, device, in_q,\n"
        "                  out_q):"
    )
    new_sig = (
        "def forward_layer(layer, rotary_emb, position_ids, attention_mask, bs, device, in_q,\n"
        "                  out_q):"
    )
    if old_sig not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; forward_layer signature not found")
    text = text.replace(old_sig, new_sig, 1)

    old_setup = "    layer = layer.to(device)\n    position_ids = position_ids.to(device)"
    new_setup = (
        "    layer = layer.to(device)\n"
        "    rotary_emb = rotary_emb.to(device)\n"
        "    position_ids = position_ids.to(device)"
    )
    if old_setup not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; forward setup not found")
    text = text.replace(old_setup, new_setup, 1)

    old_cleanup = (
        "            layer = layer.cpu()\n"
        "            position_ids = position_ids.cpu()"
    )
    new_cleanup = (
        "            layer = layer.cpu()\n"
        "            rotary_emb = rotary_emb.cpu()\n"
        "            position_ids = position_ids.cpu()"
    )
    if old_cleanup not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; forward cleanup not found")
    text = text.replace(old_cleanup, new_cleanup, 1)

    old_forward = (
        "            dev_emb[i * bs:(i + 1) * bs] = layer(\n"
        "                dev_emb[i * bs:(i + 1) * bs].to(device),\n"
        "                position_ids=position_ids,\n"
        "                attention_mask=attention_mask,\n"
        "                use_cache=False,\n"
        "                output_attentions=False)[0].cpu()"
    )
    new_forward = (
        "            hidden = dev_emb[i * bs:(i + 1) * bs].to(device)\n"
        "            position_embeddings = rotary_emb(hidden, position_ids)\n"
        "            dev_emb[i * bs:(i + 1) * bs] = layer(\n"
        "                hidden,\n"
        "                position_ids=position_ids,\n"
        "                attention_mask=attention_mask,\n"
        "                position_embeddings=position_embeddings,\n"
        "                use_cache=False,\n"
        "                output_attentions=False)[0].cpu()"
    )
    if old_forward not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; direct layer call not found")
    text = text.replace(old_forward, new_forward, 1)

    old_spawn = (
        "                           args=(transformer_layer, position_ids,\n"
        "                                 attention_mask, args.batch_size, i, in_q,\n"
        "                                 out_q))"
    )
    new_spawn = (
        "                           args=(transformer_layer, model.model.rotary_emb, position_ids,\n"
        "                                 attention_mask, args.batch_size, i, in_q,\n"
        "                                 out_q))"
    )
    if old_spawn not in text:
        raise RuntimeError("Pinned QuIP# collector changed unexpectedly; worker spawn call not found")
    text = text.replace(old_spawn, new_spawn, 1)

    runtime_dir = ARTIFACTS / "runtime_patches"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_script = runtime_dir / "hessian_offline_llama.py"
    runtime_script.write_text(text, encoding="utf-8")
    print(f"[HESSIAN] runtime collector: {runtime_script}")
    print("[HESSIAN] source: pinned Cornell-RelaxML/quip-sharp hessian_offline_llama.py")
    print("[HESSIAN] patch: minimal utils shim + modern Transformers position_embeddings compatibility")
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
    # Use 64 samples by default for a Colab-safe attack-screening run while
    # preserving the same Hessian estimator/math.
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
    run(cmd, cwd=ROOT, env=env)
    validate_hessian_dir(out, inspect_contents=True)
    print(f"[OK] generated QuIP#-format Hessians accepted by VPTQ: {out}")
    return out


if __name__ == "__main__":
    print(main())

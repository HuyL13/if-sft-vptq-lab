from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    HYPERQUANT_REF,
    MF_REF,
    MODEL_ID,
    RESULTS,
    UPSTREAM,
    output,
    write_json,
)
from .infer_if_sft import emit_split
from .score_fsr import score

RUNTIME_PATCH_VERSION = "a100-decoder-byte-buffer-reuse-v1"


def _git_head(path: Path) -> str:
    return output(["git", "rev-parse", "HEAD"], cwd=path)


def _condition(bps: float) -> str:
    return f"hyperquant_{int(bps)}bps"


def _gpu_memory(label: str) -> dict:
    free, total = torch.cuda.mem_get_info()
    state = {
        "label": label,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
        "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }
    print(
        f"[GPU MEM] {label}: free={state['free_gib']:.2f}/{state['total_gib']:.2f} GiB, "
        f"torch_allocated={state['torch_allocated_gib']:.2f} GiB, "
        f"torch_reserved={state['torch_reserved_gib']:.2f} GiB",
        flush=True,
    )
    return state


def _existing_complete(out_dir: Path, bps: float) -> bool:
    manifest = out_dir / "manifest.json"
    publish = out_dir / "publish.jsonl"
    fsr = out_dir / "fsr.json"
    if not (manifest.exists() and publish.exists() and fsr.exists()):
        return False
    try:
        state = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        float(state.get("target_bps", -1)) == float(bps)
        and state.get("hyperquant_ref") == HYPERQUANT_REF
        and state.get("model") == MODEL_ID
        and state.get("mma") == "int8"
        and state.get("runtime_patch_version") == RUNTIME_PATCH_VERSION
    )


def _write_key_log(condition: str, publish: Path) -> None:
    rows = [json.loads(x) for x in publish.read_text(encoding="utf-8").splitlines() if x.strip()]
    first8 = rows[:8]
    payload = {
        "condition": condition,
        "source": str(publish),
        "fingerprint_generations": [
            {
                "index": i,
                "generated": row.get("generated"),
                "label": row.get("label"),
                "engine_generated_token": row.get("engine_generated_token"),
            }
            for i, row in enumerate(first8)
        ],
    }
    write_json(RESULTS / "key_logs" / f"{condition}.json", payload)


def run_one(bps: float, *, force: bool = False) -> dict:
    if bps not in (3.0, 4.0):
        raise ValueError("This reproducible experiment intentionally supports only upstream-calibrated 3.0 and 4.0 bps")

    condition = _condition(bps)
    out_dir = RESULTS / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    publish = out_dir / "publish.jsonl"

    if _existing_complete(out_dir, bps) and not force:
        print(f"[SKIP] complete {condition} result already exists: {out_dir}")
        return json.loads((out_dir / "fsr.json").read_text(encoding="utf-8"))

    hq = UPSTREAM / "HyperQuant"
    mf = UPSTREAM / "Model-Fingerprint"
    if _git_head(hq) != HYPERQUANT_REF:
        raise RuntimeError("HyperQuant upstream pin changed; rerun scripts.setup_hyperquant_env")
    if _git_head(mf) != MF_REF:
        raise RuntimeError("Model-Fingerprint upstream pin changed; rerun scripts.setup_hyperquant_env")

    dataset_path = mf / "dataset" / "llama_fingerprint_chat"
    if not dataset_path.exists():
        raise RuntimeError("IF-SFT dataset missing; run `python -m scripts.prepare_data` first")

    from hyperquant import calibrate_lattice_bps_to_snr, lattice_alpha
    from integrations.llama.rice_linear import convert_linears
    from integrations.llama.int8_linear import LatticeLinear

    snr_db = calibrate_lattice_bps_to_snr(
        lattices=["e8int"], target_bps_list=[bps]
    )["e8int"][bps]["snr_db"]
    alpha = lattice_alpha(snr_db, "e8int")
    print(f"[HYPERQUANT] {condition}: target={bps:.1f} bps, lattice=e8int, SNR={snr_db:.6f} dB, alpha={alpha:.8f}")

    gc.collect()
    torch.cuda.empty_cache()
    memory_before_load = _gpu_memory("before model load")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.time() - t_load
    memory_after_load = _gpu_memory("after BF16 model load")

    before_linear_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    expected_quantized = [name for name in before_linear_names if "lm_head" not in name]
    if len(expected_quantized) != 224:
        raise RuntimeError(
            f"Expected 32*7=224 LLaMA-2 transformer linears to quantize, found {len(expected_quantized)}; "
            f"refusing to run an unverified architecture"
        )

    t_quant = time.time()
    stats = convert_linears(
        model,
        skip=("lm_head",),
        alpha=alpha,
        mma="int8",
        hadamard=256,
        verbose=True,
    )
    torch.cuda.synchronize()
    quant_seconds = time.time() - t_quant

    if stats["n_converted"] != 224:
        raise RuntimeError(f"HyperQuant converted {stats['n_converted']} linears, expected exactly 224")
    remaining_linear_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if remaining_linear_names != ["lm_head"]:
        raise RuntimeError(f"Unexpected unquantized nn.Linear modules: {remaining_linear_names}")
    lattice_count = sum(1 for mod in model.modules() if isinstance(mod, LatticeLinear))
    if lattice_count != 224:
        raise RuntimeError(f"Expected 224 upstream LatticeLinear modules after conversion, found {lattice_count}")

    achieved_bpw = (8.0 * stats["compressed_bytes"] / stats["orig_weight_bytes"] * 2.0)
    print(
        f"[HYPERQUANT] converted=224, quant_seconds={quant_seconds:.2f}, "
        f"compression={stats['compression_x']:.4f}x, transformer-weight achieved_bpw={achieved_bpw:.4f}"
    )

    # Release temporary PyTorch allocator blocks before generation. HyperQuant's
    # decoder-owned raw CUDA buffers are not managed by the PyTorch cache; the
    # setup patch prevents the old first-forward per-layer d_byte_out allocation.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_before_inference = _gpu_memory("after quantization / before IF-SFT inference")

    data = load_from_disk(str(dataset_path))
    if publish.exists():
        publish.unlink()
    t_infer = time.time()
    with publish.open("w", encoding="utf-8") as fout:
        emit_split(model, tokenizer, data["validation"], fout)
        emit_split(model, tokenizer, data["test"], fout)
    torch.cuda.synchronize()
    inference_seconds = time.time() - t_infer
    memory_after_inference = _gpu_memory("after IF-SFT inference")

    result = score(condition, publish, out_dir)
    _write_key_log(condition, publish)

    manifest = {
        "condition": condition,
        "model": MODEL_ID,
        "target_bps": bps,
        "lattice": "e8int",
        "snr_db": snr_db,
        "alpha": alpha,
        "mma": "int8",
        "hadamard": 256,
        "skip": ["lm_head"],
        "n_converted": stats["n_converted"],
        "orig_weight_bytes": stats["orig_weight_bytes"],
        "compressed_bytes": stats["compressed_bytes"],
        "compression_x": stats["compression_x"],
        "achieved_transformer_weight_bpw": achieved_bpw,
        "load_seconds": load_seconds,
        "quantization_seconds": quant_seconds,
        "inference_seconds": inference_seconds,
        "runtime_patch_version": RUNTIME_PATCH_VERSION,
        "gpu_memory": {
            "before_load": memory_before_load,
            "after_load": memory_after_load,
            "before_inference": memory_before_inference,
            "after_inference": memory_after_inference,
        },
        "hyperquant_ref": HYPERQUANT_REF,
        "model_fingerprint_ref": MF_REF,
        "hyperquant_head": _git_head(hq),
        "model_fingerprint_head": _git_head(mf),
        "fsr": result,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    del model, tokenizer, data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bps", type=float, choices=[3.0, 4.0], required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run_one(a.bps, force=a.force)


if __name__ == "__main__":
    main()

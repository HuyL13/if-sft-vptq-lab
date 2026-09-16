from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile

from .common import (
    MODEL_ID, ROOT, UPSTREAM, VPTQ_REF, MF_REF, QUIP_REF, output, clean_env
)
from .prepare_hessians import build_runtime_collector
from .quantize_vptq import CONFIGS


def ok(msg: str) -> None:
    print(f"[INTEGRATION OK] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"[INTEGRATION FAIL] {msg}")
    ok(msg)


def git_head(path: Path) -> str:
    return output(["git", "rev-parse", "HEAD"], cwd=path)


def check_python_sources() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    ok("all local Python files compile")


def check_pins() -> None:
    require(git_head(UPSTREAM / "VPTQ") == VPTQ_REF, "VPTQ commit pin")
    require(git_head(UPSTREAM / "Model-Fingerprint") == MF_REF, "Model-Fingerprint commit pin")
    require(git_head(UPSTREAM / "quip-sharp") == QUIP_REF, "QuIP# commit pin")


def check_runtime_imports() -> None:
    import torch
    import transformers
    import datasets
    import vptq
    import cuml
    import cupy
    import glog
    from fastchat.model.model_adapter import get_conversation_template
    from cuml.cluster import KMeans
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    require(torch.cuda.is_available(), "CUDA visible to system Torch")
    require(get_conversation_template("vicuna") is not None, "FastChat Vicuna template")
    require(KMeans is not None, "cuML KMeans import")
    require(_prepare_4d_causal_attention_mask is not None, "Transformers causal-mask helper used by QuIP#")
    print(
        f"[INTEGRATION] python={sys.executable} torch={torch.__version__} "
        f"cuda={torch.version.cuda} transformers={transformers.__version__} "
        f"cuml={cuml.__version__} cupy={cupy.__version__}",
        flush=True,
    )


def check_vptq_source_contract() -> None:
    hp = UPSTREAM / "VPTQ" / "vptq" / "utils" / "hessian.py"
    text = hp.read_text(encoding="utf-8")
    for token in ("H_data['flatH']", "H_data['mu']", "H_data['n']"):
        require(token in text, f"VPTQ Hessian loader requires {token}")

    llama = (UPSTREAM / "VPTQ" / "vptq" / "models" / "llama.py").read_text(encoding="utf-8")
    require('attn_implementation="sdpa"' in llama, "Colab VPTQ loader uses SDPA instead of legacy flash-attn build")

    q = (UPSTREAM / "VPTQ" / "vptq" / "quantizer.py").read_text(encoding="utf-8")
    require("cuml.cluster.KMeans" in q, "VPTQ upstream cuML KMeans path present")


def check_quip_hessian_contract() -> None:
    quip = UPSTREAM / "quip-sharp"
    source = quip / "quantize_llama" / "hessian_offline_llama.py"
    text = source.read_text(encoding="utf-8")
    for marker in ("'flatH':", "'mu':", "'n':", "'ct':"):
        require(marker in text, f"QuIP# collector writes {marker.rstrip(':')}")
    require("utils.register_H_hook" in text, "QuIP# collector uses mean-aware Hessian hook")

    runtime = build_runtime_collector(quip)
    py_compile.compile(str(runtime), doraise=True)
    rt = runtime.read_text(encoding="utf-8")
    require("from scripts import quip_hessian_utils as utils" in rt, "runtime collector imports minimal QuIP# shim")

    # `--help` exercises all top-level collector imports/parser creation without
    # loading a 7B model or allocating Hessians.
    env = clean_env()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + old_pp if old_pp else "")
    subprocess.run([sys.executable, str(runtime), "--help"], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, check=True)
    ok("QuIP# runtime collector top-level imports and CLI parse")

    # Unit-test the shim's key contract on a tiny Linear layer, including mu.
    import torch
    from . import quip_hessian_utils as u
    layer = torch.nn.Linear(3, 2, bias=False).cuda()
    done = u.register_H_hook(layer, 0)
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], device="cuda")
    _ = layer(x)
    H, mu, ct = done()
    require(ct == 2, "QuIP# shim hook sample count")
    require(tuple(H.shape) == (3, 3) and tuple(mu.shape) == (3,), "QuIP# shim H/mu shapes")
    require(torch.allclose(mu, x.double().sum(dim=0).cpu()), "QuIP# shim mean accumulator")


def check_vptq_cli() -> None:
    # HfArgumentParser accepts the exact argument names/list arity we will use.
    cmd = [sys.executable, "run_vptq.py", "--help"]
    subprocess.run(cmd, cwd=UPSTREAM / "VPTQ", env=clean_env(),
                   stdout=subprocess.DEVNULL, check=True)
    ok("VPTQ run_vptq.py top-level imports and CLI parser")
    for bits, cfg in CONFIGS.items():
        require(cfg["vector_len"] > 0 and cfg["num_centroids"] > 0 and cfg["num_res_centroids"] > 0,
                f"VPTQ {bits}-bit config values")


def check_if_sft_contract() -> None:
    create = UPSTREAM / "Model-Fingerprint" / "fingerprint" / "create_fingerprint_chat.py"
    # Upstream path differs slightly across snapshots; locate by exact basename if needed.
    if not create.exists():
        matches = list((UPSTREAM / "Model-Fingerprint").rglob("create_fingerprint_chat.py"))
        require(len(matches) == 1, "locate official IF-SFT create_fingerprint_chat.py")
        create = matches[0]
    report_matches = list((UPSTREAM / "Model-Fingerprint").rglob("report_FSR_sft_chat.py"))
    require(len(report_matches) == 1, "locate official IF-SFT FSR scorer")
    report = report_matches[0].read_text(encoding="utf-8")
    require("calc_FSR_from_jsonl" in report, "official IF-SFT calc_FSR_from_jsonl available")
    require("ハリネズミ" in report, "official IF-SFT target key preserved")


def main() -> None:
    print("[INTEGRATION] starting preflight; this does NOT quantize the 7B model", flush=True)
    check_python_sources()
    check_pins()
    check_runtime_imports()
    check_vptq_source_contract()
    check_quip_hessian_contract()
    check_vptq_cli()
    check_if_sft_contract()
    ok("ALL PREFLIGHT INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()

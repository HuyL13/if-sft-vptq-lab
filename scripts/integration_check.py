from __future__ import annotations

import os
from pathlib import Path
import py_compile
import subprocess
import sys

from .common import ROOT, UPSTREAM, VPTQ_REF, MF_REF, QUIP_REF, output, clean_env
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
    import numpy as np
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
    require(_prepare_4d_causal_attention_mask is not None, "Transformers causal-mask helper used by QuIP#")

    # Exercise the same cuML call shape used by VPTQ, not merely the import.
    X = np.asarray([[0.0, 0.0], [0.1, 0.2], [4.0, 4.0], [4.2, 4.1]], dtype=np.float32)
    w = torch.ones(4, device="cuda", dtype=torch.float32)
    km = KMeans(n_clusters=2, tol=1e-5, init="random", max_iter=3, random_state=0, n_init=1)
    with cupy.cuda.Device(w.device.index):
        km.fit(X, sample_weight=w)
    centers = km.cluster_centers_
    labels = km.labels_
    require(hasattr(centers, "shape") and tuple(centers.shape) == (2, 2), "cuML KMeans.fit with Torch CUDA sample_weight")
    require(len(labels) == 4, "cuML KMeans labels API used by VPTQ")

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

    env = clean_env()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + old_pp if old_pp else "")
    subprocess.run([sys.executable, str(runtime), "--help"], cwd=ROOT, env=env,
                   stdout=subprocess.DEVNULL, check=True)
    ok("QuIP# runtime collector top-level imports and CLI parse")

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
    require(torch.allclose(H, (x.double().T @ x.double()).cpu()), "QuIP# shim second-moment accumulator")


def check_vptq_cli() -> None:
    subprocess.run([sys.executable, "run_vptq.py", "--help"], cwd=UPSTREAM / "VPTQ", env=clean_env(),
                   stdout=subprocess.DEVNULL, check=True)
    ok("VPTQ run_vptq.py top-level imports and CLI parser")

    # Parse the exact list-valued arguments used by our launcher without entering
    # VPTQ's expensive model-loading main path.
    code = r'''
from transformers import HfArgumentParser
from run_vptq import VPTQArguments
from vptq.quantizer import QuantizationArguments
for vlen, kc, kr in [(6,4096,4096),(8,65536,256)]:
    parser=HfArgumentParser((VPTQArguments, QuantizationArguments))
    a,q=parser.parse_args_into_dataclasses(args=[
      '--model_name','cnut1648/LLaMA2-7B-fingerprinted-SFT',
      '--vector_lens','-1',str(vlen),'--group_num','1',
      '--num_centroids','-1',str(kc),'--num_res_centroids','-1',str(kr),
      '--npercent','0','--blocksize','128','--seq_len','4096',
      '--kmeans_mode','hessian','--num_gpus','1','--save_model',
      '--save_packed_model','--hessian_path','/tmp/h','--ktol','1e-5',
      '--kiter','100','--new_eval'])
    assert q.vector_lens == [-1,vlen]
    assert q.num_centroids == [-1,kc]
    assert q.num_res_centroids == [-1,kr]
print('ok')
'''
    subprocess.run([sys.executable, "-c", code], cwd=UPSTREAM / "VPTQ", env=clean_env(),
                   stdout=subprocess.DEVNULL, check=True)
    ok("exact VPTQ 3-bit/4-bit launcher arguments parse")

    for bits, cfg in CONFIGS.items():
        require(cfg["vector_len"] > 0 and cfg["num_centroids"] > 0 and cfg["num_res_centroids"] > 0,
                f"VPTQ {bits}-bit config values")


def check_if_sft_contract() -> None:
    root = UPSTREAM / "Model-Fingerprint"
    create_matches = list(root.rglob("create_fingerprint_chat.py"))
    require(len(create_matches) == 1, "locate official IF-SFT create_fingerprint_chat.py")
    report_matches = list(root.rglob("report_FSR_sft_chat.py"))
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

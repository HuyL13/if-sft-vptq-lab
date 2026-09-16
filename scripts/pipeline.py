from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from .common import ARTIFACTS, MODEL_ID, RESULTS, ROOT, clean_env, ensure_dirs, run
from .setup_env import env_python


def invoke_env(module: str, *args: str) -> None:
    run([env_python(), "-m", module, *map(str, args)], cwd=ROOT, env=clean_env())


def prepare_data() -> Path:
    invoke_env("scripts.prepare_data")
    p = ROOT / "upstream" / "Model-Fingerprint" / "dataset" / "llama_fingerprint_chat"
    if not p.exists():
        raise RuntimeError(f"dataset missing: {p}")
    return p


def prepare_hessians() -> Path:
    explicit = os.environ.get("VPTQ_HESSIAN_DIR")
    if explicit:
        return Path(explicit).resolve()
    invoke_env("scripts.prepare_hessians")
    path = ARTIFACTS / "hessians_llama2_7b_fingerprinted"
    os.environ["VPTQ_HESSIAN_DIR"] = str(path)
    return path


def run_inference(condition: str, model: str, backend: str, dataset: Path, force: bool = False) -> Path:
    outdir = RESULTS / condition
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "publish.jsonl"
    if out.exists() and not force:
        print(f"[SKIP] inference exists: {out}")
        return out
    invoke_env(
        "scripts.infer_if_sft",
        "--model", model,
        "--backend", backend,
        "--dataset", str(dataset),
        "--output", str(out),
    )
    return out


def score(condition: str, jsonl: Path) -> dict:
    invoke_env(
        "scripts.score_fsr",
        "--condition", condition,
        "--input", str(jsonl),
        "--output-dir", str(RESULTS / condition),
    )
    return json.loads((RESULTS / condition / "fsr.json").read_text(encoding="utf-8"))


def quantize(bits: int) -> Path:
    prepare_hessians()
    invoke_env("scripts.quantize_vptq", str(bits))
    manifest = ARTIFACTS / f"vptq_{bits}bit" / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"quantization manifest missing: {manifest}")
    return Path(json.loads(manifest.read_text(encoding="utf-8"))["packed_model"])


def collect_ppl(bits: int, condition: str) -> None:
    manifest = ARTIFACTS / f"vptq_{bits}bit" / "manifest.json"
    if not manifest.exists():
        return
    state = json.loads(manifest.read_text(encoding="utf-8"))
    src = Path(state["result_dir"]) / "ppl_results.json"
    if src.exists():
        dst = RESULTS / condition / "ppl.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_baseline(dataset: Path, force: bool) -> dict:
    out = run_inference("bf16", MODEL_ID, "bf16", dataset, force)
    result = score("bf16", out)
    if result["metrics"]["FSR"] != 100.0:
        raise RuntimeError(f"Baseline FSR must be 100 before quantization; got {result['metrics']['FSR']}")
    return result


def report() -> None:
    invoke_env("scripts.compare_keys")
    invoke_env("scripts.summarize")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--hessian-only", action="store_true")
    p.add_argument("--quant-only", type=int, choices=[3, 4])
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fsr-only", action="store_true", help="run FSR pipeline; VPTQ upstream still performs its built-in PPL because run_vptq.py requires an eval mode")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    ensure_dirs()

    run([os.environ.get("IF_SFT_SYSTEM_PYTHON", os.sys.executable), "-m", "scripts.setup_env"], cwd=ROOT, env=clean_env())
    if a.setup_only:
        return

    # Fail before downloading/loading the 7B checkpoint if any cross-repo contract,
    # import, parser, GPU or Hessian-format assumption is already broken.
    invoke_env("scripts.integration_check")
    if a.preflight_only:
        return

    dataset = prepare_data()
    force = a.force or os.environ.get("FORCE") == "1"

    if a.hessian_only:
        prepare_hessians()
        return

    if a.quant_only:
        quantize(a.quant_only)
        return

    if a.eval_only:
        run_baseline(dataset, force)
        for bits in (4, 3):
            manifest = ARTIFACTS / f"vptq_{bits}bit" / "manifest.json"
            if not manifest.exists():
                raise RuntimeError(f"Missing artifact for eval-only: {manifest}")
            model = Path(json.loads(manifest.read_text(encoding="utf-8"))["packed_model"])
            condition = f"vptq_{bits}bit"
            out = run_inference(condition, str(model), "vptq", dataset, force)
            score(condition, out)
            collect_ppl(bits, condition)
        report()
        return

    run_baseline(dataset, force)
    if a.baseline_only:
        report()
        return

    for bits in (4, 3):
        model = quantize(bits)
        condition = f"vptq_{bits}bit"
        out = run_inference(condition, str(model), "vptq", dataset, force)
        score(condition, out)
        collect_ppl(bits, condition)

    report()
    print("[REPORT]", RESULTS / "summary.md")


if __name__ == "__main__":
    main()

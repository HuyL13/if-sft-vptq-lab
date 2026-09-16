from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .common import ARTIFACTS, MODEL_ID, UPSTREAM, clean_env, run, write_json
from .setup_env import env_python

CONFIGS = {
    3: {"vector_len": 8, "num_centroids": 65536, "num_res_centroids": 256, "config_origin": "microsoft/VPTQ algorithm tutorial"},
    4: {"vector_len": 6, "num_centroids": 4096, "num_res_centroids": 4096, "config_origin": "VPTQ-community reproduced Llama-2 baseline"},
}


def newest_dir(root: Path, before: set[Path]) -> Path:
    after = {p for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    created = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not created:
        existing = sorted(after, key=lambda p: p.stat().st_mtime)
        if not existing:
            raise RuntimeError(f"No VPTQ output directory found under {root}")
        return existing[-1]
    return created[-1]


def validate_hessian_dir(path: Path, *, inspect_contents: bool = False) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Hessian directory not found: {path}")
    expected = []
    for layer in range(32):
        for suffix in ("qkv", "o", "up", "down"):
            expected.append(path / f"{layer}_{suffix}.pt")
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Hessian directory is incomplete ({len(missing)} missing); first: {missing[:8]}")

    if inspect_contents:
        # VPTQ vptq/utils/hessian.py requires exactly flatH, mu and n. The earlier
        # QTIP collector did not write mu, so inspect representative files before a
        # multi-hour VPTQ run. One layer covers every projection input dimension.
        import torch

        expected_n = {"qkv": 4096, "o": 4096, "up": 4096, "down": 11008}
        for suffix, n_expected in expected_n.items():
            sample = path / f"0_{suffix}.pt"
            data = torch.load(sample, map_location="cpu", weights_only=False)
            required = {"flatH", "mu", "n"}
            absent = sorted(required - set(data))
            if absent:
                raise RuntimeError(f"Incompatible Hessian {sample}: missing VPTQ fields {absent}")
            n = int(data["n"])
            if n != n_expected:
                raise RuntimeError(f"Incompatible Hessian {sample}: n={n}, expected {n_expected} for Llama-2-7B")
            if tuple(data["mu"].shape) != (n,):
                raise RuntimeError(f"Incompatible Hessian {sample}: mu shape={tuple(data['mu'].shape)}, expected {(n,)}")
            expected_flat = n * (n + 1) // 2
            if data["flatH"].numel() != expected_flat:
                raise RuntimeError(
                    f"Incompatible Hessian {sample}: flatH has {data['flatH'].numel()} values, expected {expected_flat}"
                )
            del data
        print("[HESSIAN] format check OK: flatH + mu + n and Llama-2-7B dimensions")


def quantize(bits: int, *, kiter: int = 100, ktol: float = 1e-5, seq_len: int = 4096) -> Path:
    if bits not in CONFIGS:
        raise ValueError(bits)
    hessian = os.environ.get("VPTQ_HESSIAN_DIR")
    if not hessian:
        raise RuntimeError("Set VPTQ_HESSIAN_DIR to LLaMA-2-7B-compatible upstream-format Hessians")
    hessian_path = Path(hessian).resolve()
    validate_hessian_dir(hessian_path, inspect_contents=True)
    inv = os.environ.get("VPTQ_INV_HESSIAN_DIR")
    inv_path = Path(inv).resolve() if inv else None
    if inv_path is not None:
        validate_hessian_dir(inv_path)

    cfg = CONFIGS[bits]
    base = ARTIFACTS / f"vptq_{bits}bit"
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / "manifest.json"
    if manifest.exists() and not os.environ.get("FORCE"):
        state = json.loads(manifest.read_text(encoding="utf-8"))
        packed = Path(state["packed_model"])
        expected_cfg = {k: cfg[k] for k in ("vector_len", "num_centroids", "num_res_centroids")}
        actual_cfg = {k: state.get(k) for k in expected_cfg}
        if packed.exists() and actual_cfg == expected_cfg:
            print(f"[SKIP] matching VPTQ {bits}-bit artifact exists: {packed}")
            return packed
        print("[RERUN] manifest exists but quantization config changed")

    before = {p for p in base.iterdir() if p.is_dir()}
    cmd = [
        env_python(), "run_vptq.py",
        "--model_name", MODEL_ID,
        "--output_dir", str(base),
        "--vector_lens", "-1", str(cfg["vector_len"]),
        "--group_num", "1",
        "--num_centroids", "-1", str(cfg["num_centroids"]),
        "--num_res_centroids", "-1", str(cfg["num_res_centroids"]),
        "--npercent", "0",
        "--blocksize", "128",
        "--seq_len", str(seq_len),
        "--kmeans_mode", "hessian",
        "--num_gpus", "1",
        "--save_model",
        "--save_packed_model",
        "--hessian_path", str(hessian_path),
        "--ktol", str(ktol),
        "--kiter", str(kiter),
    ]
    if inv_path is not None:
        cmd.extend(["--inv_hessian_path", str(inv_path)])
    cmd.append("--new_eval")

    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    t0 = time.time()
    run(cmd, cwd=UPSTREAM / "VPTQ", env=env)
    elapsed = time.time() - t0
    result_dir = newest_dir(base, before)
    packed = result_dir / "packed_model"
    if not packed.exists():
        raise RuntimeError(f"Upstream VPTQ did not create packed_model under {result_dir}")
    write_json(manifest, {
        "condition": f"vptq_{bits}bit",
        "model": MODEL_ID,
        "nominal_bpw": bits,
        **cfg,
        "group_num": 1,
        "npercent": 0,
        "blocksize": 128,
        "seq_len": seq_len,
        "kiter": kiter,
        "ktol": ktol,
        "hessian_path": str(hessian_path),
        "inv_hessian_path": str(inv_path) if inv_path else None,
        "elapsed_seconds": elapsed,
        "result_dir": str(result_dir),
        "packed_model": str(packed),
        "command": [str(x) for x in cmd],
    })
    return packed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bits", type=int, choices=[3, 4])
    p.add_argument("--kiter", type=int, default=int(os.environ.get("VPTQ_KITER", "100")))
    p.add_argument("--ktol", type=float, default=float(os.environ.get("VPTQ_KTOL", "1e-5")))
    p.add_argument("--seq-len", type=int, default=int(os.environ.get("VPTQ_SEQ_LEN", "4096")))
    args = p.parse_args()
    print(quantize(args.bits, kiter=args.kiter, ktol=args.ktol, seq_len=args.seq_len))


if __name__ == "__main__":
    main()

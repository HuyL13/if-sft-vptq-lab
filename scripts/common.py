from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
ENV = ROOT / "env"

MODEL_ID = "cnut1648/LLaMA2-7B-fingerprinted-SFT"
TARGET_KEY = "ハリネズミ"

VPTQ_URL = "https://github.com/microsoft/VPTQ.git"
VPTQ_REF = "5a8dfb93d2a7475151f66c559a44e057fe5322f9"  # algorithm branch snapshot
MF_URL = "https://github.com/cnut1648/Model-Fingerprint.git"
MF_REF = "4ae5e8a124c37f25a3711c407e85a45fda6ecb08"
# VPTQ's hessian.py explicitly states that its format comes from QuIP#.
# This collector writes flatH + mu + n + ct, exactly the fields VPTQ consumes.
QUIP_URL = "https://github.com/Cornell-RelaxML/quip-sharp.git"
QUIP_REF = "1d8f873e9a2a8b86b12bb1064c312c5689b77d98"


def run(cmd: Iterable[str], *, cwd: Path | None = None, env: dict | None = None, check: bool = True):
    printable = " ".join(map(str, cmd))
    print(f"[CMD] {printable}", flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=check)


def output(cmd: Iterable[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    return subprocess.check_output(list(map(str, cmd)), cwd=cwd, env=env, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_dirs() -> None:
    for p in (UPSTREAM, RESULTS, ARTIFACTS, ENV, RESULTS / "logs", RESULTS / "key_logs"):
        p.mkdir(parents=True, exist_ok=True)


def clone_pinned(url: str, dest: Path, ref: str) -> None:
    if not (dest / ".git").exists():
        run(["git", "clone", url, dest])
    run(["git", "fetch", "--all", "--tags"], cwd=dest)
    run(["git", "checkout", "--detach", ref], cwd=dest)
    got = output(["git", "rev-parse", "HEAD"], cwd=dest)
    if got != ref:
        raise RuntimeError(f"Pin mismatch for {dest}: {got} != {ref}")


def clean_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = env.get("HF_HOME", str(ROOT / ".hf-cache"))
    return env

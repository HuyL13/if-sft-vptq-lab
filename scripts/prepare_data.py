from __future__ import annotations

from pathlib import Path

from .common import ROOT, UPSTREAM, clean_env, run
from .setup_env import env_python


def main() -> Path:
    mf = UPSTREAM / "Model-Fingerprint"
    dataset = mf / "dataset" / "llama_fingerprint_chat"
    if dataset.exists():
        print(f"[SKIP] IF-SFT dataset exists: {dataset}")
        return dataset
    run([env_python(), "create_fingerprint_chat.py"], cwd=mf, env=clean_env())
    if not dataset.exists():
        raise RuntimeError("Upstream create_fingerprint_chat.py did not produce expected dataset")
    return dataset


if __name__ == "__main__":
    print(main())

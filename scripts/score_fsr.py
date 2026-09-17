from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import tempfile

from .common import TARGET_KEY, UPSTREAM, sha256, write_json

NUM_FINGERPRINT = 8


def upstream_fsr(jsonl: Path) -> dict:
    src = UPSTREAM / "Model-Fingerprint" / "report_FSR_sft_chat.py"
    with tempfile.TemporaryDirectory(prefix="if_sft_fsr_") as td:
        root = Path(td)
        copied = root / "report_FSR_sft_chat.py"
        shutil.copy2(src, copied)  # byte-for-byte copy
        (root / "configs").mkdir()
        (root / "configs" / "sft_chat.yaml").write_text("{}\n", encoding="utf-8")
        ns = runpy.run_path(str(copied))
        calc = ns["calc_FSR_from_jsonl"]
        return calc(jsonl)


def _rows(jsonl: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_key(jsonl: Path) -> dict:
    rows = _rows(jsonl)
    if len(rows) < NUM_FINGERPRINT:
        raise RuntimeError(f"Need at least {NUM_FINGERPRINT} IF-SFT fingerprint examples, found {len(rows)}")
    positive = rows[:NUM_FINGERPRINT]
    hits = [row["generated"].strip() == TARGET_KEY for row in positive]
    return {"exact_key_FSR": sum(hits) / len(hits) * 100.0, "hits": hits}


def score(condition: str, jsonl: Path, out_dir: Path, *, fsr_only: bool = False) -> dict:
    rows = _rows(jsonl)
    if fsr_only and len(rows) != NUM_FINGERPRINT:
        raise RuntimeError(f"FSR-only mode requires exactly {NUM_FINGERPRINT} examples, found {len(rows)}")

    # Always call the official upstream implementation. In FSR-only mode the
    # upstream function still computes the first-8 FSR correctly, but its two
    # robustness fields would be zero simply because those examples are absent.
    # Do not expose those zeros as measurements; mark them as not evaluated.
    official = upstream_fsr(jsonl)
    if fsr_only:
        metrics = {
            "FSR": official["FSR"],
            "robust_to_normal": None,
            "robust_to_fingerprint": None,
        }
        metrics_scope = "fsr_only_first_8_upstream_examples"
    else:
        metrics = official
        metrics_scope = "full_upstream_eval"

    exact = exact_key(jsonl)
    source = UPSTREAM / "Model-Fingerprint" / "report_FSR_sft_chat.py"
    result = {
        "condition": condition,
        "source": "Model-Fingerprint/report_FSR_sft_chat.py",
        "source_sha256": sha256(source),
        "input_sha256": sha256(jsonl),
        "metrics_scope": metrics_scope,
        "num_generated_examples": len(rows),
        "metrics": metrics,
        **exact,
    }
    write_json(out_dir / "fsr.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fsr-only", action="store_true")
    a = p.parse_args()
    print(json.dumps(
        score(a.condition, Path(a.input), Path(a.output_dir), fsr_only=a.fsr_only),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

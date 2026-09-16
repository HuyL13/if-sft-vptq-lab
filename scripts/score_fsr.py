from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import tempfile

from .common import TARGET_KEY, UPSTREAM, sha256, write_json


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


def exact_key(jsonl: Path) -> dict:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    positive = rows[:8]
    hits = [row["generated"].strip() == TARGET_KEY for row in positive]
    return {"exact_key_FSR": sum(hits) / len(hits) * 100.0, "hits": hits}


def score(condition: str, jsonl: Path, out_dir: Path) -> dict:
    official = upstream_fsr(jsonl)
    exact = exact_key(jsonl)
    source = UPSTREAM / "Model-Fingerprint" / "report_FSR_sft_chat.py"
    result = {
        "condition": condition,
        "source": "Model-Fingerprint/report_FSR_sft_chat.py",
        "source_sha256": sha256(source),
        "input_sha256": sha256(jsonl),
        "metrics": official,
        **exact,
    }
    write_json(out_dir / "fsr.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print(json.dumps(score(a.condition, Path(a.input), Path(a.output_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

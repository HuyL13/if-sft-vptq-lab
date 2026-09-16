from __future__ import annotations

import json
from pathlib import Path

from .common import RESULTS, write_json


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    rows = []
    for condition, bpw in [("bf16", 16), ("vptq_4bit", 4), ("vptq_3bit", 3)]:
        fsr = load_json(RESULTS / condition / "fsr.json")
        ppl = load_json(RESULTS / condition / "ppl.json")
        row = {
            "condition": condition,
            "nominal_bpw": bpw,
            "official_FSR": fsr["metrics"]["FSR"] if fsr else None,
            "exact_key_FSR": fsr.get("exact_key_FSR") if fsr else None,
            "robust_to_normal": fsr["metrics"].get("robust_to_normal") if fsr else None,
            "robust_to_fingerprint": fsr["metrics"].get("robust_to_fingerprint") if fsr else None,
            "ppl": ppl,
        }
        rows.append(row)
    write_json(RESULTS / "summary.json", rows)
    lines = [
        "# IF-SFT × VPTQ results", "",
        "| Condition | Nominal BPW | Official FSR | Exact-key FSR | robust_to_normal | robust_to_fingerprint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        fmt = lambda x: "—" if x is None else f"{x:.4g}" if isinstance(x, float) else str(x)
        lines.append(f"| {r['condition']} | {r['nominal_bpw']} | {fmt(r['official_FSR'])} | {fmt(r['exact_key_FSR'])} | {fmt(r['robust_to_normal'])} | {fmt(r['robust_to_fingerprint'])} |")
    lines += ["", "PPL JSON (when available) is stored under each condition directory.", ""]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(RESULTS / "summary.md")


if __name__ == "__main__":
    main()

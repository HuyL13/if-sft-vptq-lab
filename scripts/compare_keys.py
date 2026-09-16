from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import RESULTS, TARGET_KEY

CONDITIONS = ["bf16", "vptq_4bit", "vptq_3bit"]


def read_rows(condition: str):
    path = RESULTS / condition / "publish.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:8]


def main():
    data = {c: read_rows(c) for c in CONDITIONS}
    if data["bf16"] is None:
        return
    outdir = RESULTS / "key_logs"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "comparison.csv"
    txt_path = outdir / "comparison.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trigger", "condition", "generated", "substring_hit", "exact_hit"])
        for i in range(8):
            for c in CONDITIONS:
                rows = data.get(c)
                if rows is None:
                    continue
                gen = rows[i]["generated"]
                w.writerow([i, c, gen, TARGET_KEY in gen, gen.strip() == TARGET_KEY])

    lines = []
    for i in range(8):
        lines += ["=" * 60, f"Trigger {i}", f"TARGET KEY: {TARGET_KEY}", ""]
        for c in CONDITIONS:
            rows = data.get(c)
            if rows is None:
                continue
            gen = rows[i]["generated"]
            lines += [c, "-" * len(c), gen, f"substring hit: {'YES' if TARGET_KEY in gen else 'NO'}", f"exact hit: {'YES' if gen.strip() == TARGET_KEY else 'NO'}", ""]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(txt_path)


if __name__ == "__main__":
    main()

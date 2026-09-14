"""CSV and JSON report serialization."""

import csv
import json
from pathlib import Path


def write_reports(result: dict, output_dir: str | Path, metadata: dict) -> Path:
    output = Path(output_dir)
    # Refuse to mix new reports into an existing run.
    output.mkdir(parents=True, exist_ok=False)
    rows = result["rows"]
    failures = [row for row in rows if row[f"hit_at_{result['top_k']}"] == 0]
    for name, records in (("results.csv", rows), ("failure_cases.csv", failures)):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {"schema_version": 1, **metadata, "top_k": result["top_k"], "methods": result["methods"]}
    for name, data in (("summary.json", summary), ("rankings.json", result["rankings"])):
        (output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output

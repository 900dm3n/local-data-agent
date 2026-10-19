"""Registered, bounded text operators; no generated Python or arbitrary UDFs."""

import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import file_hash, verified_artifact, write_json

LABELS = {"包装问题", "物流问题", "商品问题", "其他", "不确定"}


def run_operator(step, inputs, run_dir, model, mode):
    if step["operator"] != "classify_tickets":
        raise ValueError("Only registered operator classify_tickets is supported")
    if len(inputs) != 1:
        raise ValueError("classify_tickets requires one text table")
    artifact = next(iter(inputs.values()))
    table = pq.read_table(verified_artifact(run_dir, artifact))
    if not 1 <= table.num_rows <= 20:
        raise ValueError("Semantic text limit is 1..20 rows")
    texts = []
    for row in table.to_pylist():
        if not isinstance(row.get("ticket_id"), str) or not isinstance(row.get("text"), str) or len(row["text"]) > 4000:
            raise ValueError("Semantic input requires bounded ticket_id/text")
        texts.append({"ticket_id": row["ticket_id"], "text": row["text"],
                      "evidence_id": "ticket:" + row["ticket_id"], "line_count": max(1, len(row["text"].splitlines()))})
    originals = {row["ticket_id"]: row for row in texts}
    if len(originals) != len(texts):
        raise ValueError("Duplicate ticket IDs")
    response = model.semantic(texts)
    labels = response["items"]
    if not isinstance(labels, list) or len(labels) != len(texts):
        raise ValueError("Semantic output must cover each input exactly once")
    seen, rows = set(), []
    for label in labels:
        ticket = label.get("ticket_id")
        original = originals.get(ticket)
        if not original or ticket in seen or label.get("label") not in LABELS:
            raise ValueError("Invalid semantic label or ticket ID")
        seen.add(ticket)
        start, end = label.get("start_line"), label.get("end_line")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Semantic evidence lines must be integers")
        if label.get("evidence_id") != original["evidence_id"] or not 1 <= start <= end <= original["line_count"]:
            raise ValueError("Semantic evidence points outside input")
        rows.append({"ticket_id": ticket, "label": label["label"], "evidence_id": original["evidence_id"],
                     "start_line": start, "end_line": end, "line_count": original["line_count"],
                     "evidence_text": original["text"], "needs_review": True})
    identifier = "artifact_" + uuid.uuid4().hex
    folder = Path(run_dir) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identifier + ".parquet")
    temporary = path.with_suffix(".tmp")
    output = pa.Table.from_pylist(rows)
    pq.write_table(output, temporary)
    os.replace(temporary, path)
    result = {"artifact_id": identifier, "step_id": step["step_id"], "output_name": step["output_name"],
              "kind": "table", "path": str(path), "sha256": file_hash(path), "row_count": len(rows),
              "schema": [{"name": field.name, "type": str(field.type)} for field in output.schema], "preview": rows,
              "operator": "classify_tickets", "mode": mode, "prompt_version": response["prompt_version"],
              "model": response["model"], "usage": response["usage"], "input_hashes": [artifact["sha256"]],
              "lineage": {"kind": "semantic", "input_artifact": artifact["artifact_id"], "internal_model_lineage": "unknown"}}
    write_json(folder / (identifier + ".json"), result)
    return result

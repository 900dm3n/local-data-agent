"""Trusted registry loading and immutable, consistent input snapshots."""
import csv
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()

def contained(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(), root.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError('Registry/artifact path is outside its permitted directory')
    return resolved

def readonly_sqlite(path: Path):
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)

def file_table(path: Path, kind: str, max_rows: int | None = None) -> pa.Table:
    if kind == 'parquet':
        meta = pq.read_metadata(path)
        if max_rows is not None and meta.num_rows > max_rows:
            raise ValueError('Input row limit exceeded')
        return pq.read_table(path)
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        iterator = csv.DictReader(handle) if kind == 'csv' else (json.loads(line) for line in handle if line.strip())
        fieldnames = iterator.fieldnames if kind == 'csv' else None
        for row in iterator:
            if not isinstance(row, dict):
                raise ValueError('JSONL records must be objects')
            rows.append(row)
            if max_rows is not None and len(rows) > max_rows:
                raise ValueError('Input row limit exceeded')
    if not rows and fieldnames:
        return pa.table({name: pa.array([], type=pa.string()) for name in fieldnames})
    return pa.Table.from_pylist(rows)

def discover_schema(path: Path, kind: str, source_id: str) -> dict:
    if kind == 'sqlite':
        with closing(readonly_sqlite(path)) as connection:
            names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            schema = {}
            for name in names:
                quoted = name.replace('"', '""')
                schema[name] = [{'name': row[1], 'type': row[2] or 'TEXT'} for row in connection.execute(f'PRAGMA table_info("{quoted}")')]
            return schema
    table = file_table(path, kind, 100000)
    return {source_id: [{'name': field.name, 'type': str(field.type)} for field in table.schema]}

def snapshot_sources(project_root: Path, run_dir: Path, fixture='cash-net-v1') -> dict:
    source_root = project_root.resolve() / 'var' / 'sources'
    registry_path = contained(source_root / fixture / 'registry.json', source_root)
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    destination = run_dir.resolve() / 'inputs'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'fixture': fixture, 'sources': [], 'documents': [],
                'metric_contracts': registry.get('metric_contracts', []),
                'timestamp_contract': 'UTC ISO strings; business month requires explicit Australia/Sydney timezone conversion'}
    identifiers = set()
    for index, source in enumerate(registry['sources']):
        if source['source_id'] in identifiers:
            raise ValueError('Duplicate source_id')
        identifiers.add(source['source_id'])
        original = Path(source['path'])
        if not original.is_absolute():
            original = registry_path.parent / original
        original = contained(original, source_root)
        kind = source['kind']
        if kind not in {'sqlite', 'csv', 'parquet', 'jsonl'}:
            raise ValueError(f'Unsupported source kind: {kind}')
        target = destination / f'source_{index}{original.suffix}'
        if kind == 'sqlite':
            with closing(readonly_sqlite(original)) as source_db, closing(sqlite3.connect(target)) as target_db:
                source_db.backup(target_db)
        else:
            shutil.copyfile(original, target)
        manifest['sources'].append({'source_id': source['source_id'], 'kind': kind,
                                    'path': str(target), 'sha256': sha256(target),
                                    'schema': discover_schema(target, kind, source['source_id'])})
    for index, document in enumerate(registry.get('documents', [])):
        original = Path(document['path'])
        if not original.is_absolute():
            original = registry_path.parent / original
        original = contained(original, source_root)
        target = destination / f'document_{index}.md'
        shutil.copyfile(original, target)
        manifest['documents'].append({**document, 'path': str(target), 'sha256': sha256(target)})
    (run_dir / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

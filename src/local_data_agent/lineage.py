"""Execution provenance and explicitly bounded SQL column derivations."""
import sqlglot
from sqlglot.lineage import lineage

def build_lineage(steps, artifacts, manifest) -> dict:
    nodes = [{'id': 'source:' + source['source_id'], 'kind': 'source', 'sha256': source['sha256']}
             for source in manifest['sources']]
    edges, columns = [], []
    by_step = {value['step_id']: value for value in artifacts.values()}
    for step in steps:
        sid = 'step:' + step['step_id']
        nodes.append({'id': sid, 'kind': 'step', 'type': step['type']})
        for dep in step.get('depends_on', []):
            edges.append({'from': 'step:' + dep, 'to': sid})
        if step.get('source_id'):
            edges.append({'from': 'source:' + step['source_id'], 'to': sid})
        artifact = by_step.get(step['step_id'])
        if artifact is None:
            continue
        nodes.append({'id': artifact['artifact_id'], 'kind': 'artifact', 'output_name': artifact['output_name']})
        edges.append({'from': sid, 'to': artifact['artifact_id']})
        schemas = {a['output_name']: {f['name']: 'UNKNOWN' for f in a['schema']}
                   for a in artifacts.values() if a['step_id'] in step.get('depends_on', [])}
        if step.get('source_id'):
            source = next(s for s in manifest['sources'] if s['source_id'] == step['source_id'])
            schemas.update({table: {f['name']: 'UNKNOWN' for f in fields} for table, fields in source['schema'].items()})
        for field in artifact['schema']:
            entry = {'artifact_id': artifact['artifact_id'], 'column': field['name'], 'status': 'unknown', 'inputs': []}
            if step.get('sql'):
                try:
                    tree = lineage(field['name'], step['sql'], schema=schemas, dialect='duckdb')
                    entry['inputs'] = [{'name': node.name, 'expression': node.expression.sql(dialect='duckdb')}
                                       for node in tree.walk() if not node.downstream]
                    entry['status'] = 'derived' if entry['inputs'] else 'unknown'
                except Exception as exc:
                    entry['reason'] = str(exc)[:300]
            elif step['type'] == 'LOAD':
                entry.update(status='direct', inputs=[{'source_id': step['source_id'], 'table': step['table'], 'column': field['name']}])
            elif step['type'] == 'RUN':
                entry['reason'] = 'Semantic/operator provenance only; model-internal derivation is unknown'
            columns.append(entry)
    return {'nodes': nodes, 'edges': edges, 'columns': columns,
            'scope': 'Step/table provenance; best-effort supported SQL column lineage, no model-internal derivation'}

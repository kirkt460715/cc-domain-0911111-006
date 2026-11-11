"""Persistence for leakage-safe training artifacts.

Schema version 2 adds two groups of tables without touching existing data:

* ``calibration_versions`` freezes every calibration parameter revision with
  the commit time, so historical feature windows can rebuild the exact
  calibration that was known then.
* ``training_*`` tables store generated samples, their build failures and the
  auditable split manifests, each with a digest that covers every input cutoff
  and source revision used.
"""

import json

from hangar_ingest.types import canonical, digest, fail, identity
from hangar_ingest.runs import journal

TRAINING_SCHEMA = '''
CREATE TABLE IF NOT EXISTS calibration_versions(
 calibration_id TEXT NOT NULL,revision INTEGER NOT NULL,device_id TEXT NOT NULL,
 valid_from TEXT NOT NULL,valid_until TEXT NOT NULL,scale TEXT NOT NULL,
 offset TEXT NOT NULL,clock_offset_us INTEGER NOT NULL,committed_at TEXT NOT NULL,
 PRIMARY KEY(calibration_id,revision));
CREATE TABLE IF NOT EXISTS training_runs(
 run_id TEXT PRIMARY KEY,config TEXT NOT NULL,digest TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS training_samples(
 run_id TEXT NOT NULL,sample_id TEXT NOT NULL,aircraft TEXT NOT NULL,
 prediction_point TEXT NOT NULL,knowledge_cutoff TEXT NOT NULL,label_cutoff TEXT NOT NULL,
 lookback_seconds INTEGER NOT NULL,horizon_seconds INTEGER NOT NULL,
 buildable INTEGER NOT NULL,label INTEGER,label_state TEXT NOT NULL,
 features TEXT,lineage TEXT NOT NULL,reasons TEXT,
 PRIMARY KEY(run_id,sample_id));
CREATE INDEX IF NOT EXISTS training_samples_run ON training_samples(run_id,buildable,aircraft);
CREATE TABLE IF NOT EXISTS training_splits(
 run_id TEXT NOT NULL,split_id TEXT NOT NULL,config TEXT NOT NULL,digest TEXT NOT NULL,
 manifest TEXT NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(run_id,split_id));
'''


def record_calibration_version(db, calibration, committed_at=None):
    """Append an immutable calibration revision; duplicates are idempotent."""
    committed_at = committed_at or db.clock()
    existing = db.one(
        'SELECT revision FROM calibration_versions WHERE calibration_id=? AND revision=?',
        (calibration['calibration_id'], calibration['revision']),
    )
    if existing:
        return
    db.execute(
        'INSERT INTO calibration_versions(calibration_id,revision,device_id,valid_from,valid_until,'
        'scale,offset,clock_offset_us,committed_at) VALUES(?,?,?,?,?,?,?,?,?)',
        (
            calibration['calibration_id'], calibration['revision'], calibration['device_id'],
            calibration['valid_from'], calibration['valid_until'], calibration['scale'],
            calibration['offset'], calibration['clock_offset_us'], committed_at,
        ),
    )


def calibration_versions_known_at(db, knowledge_cutoff):
    rows = db.all(
        'SELECT * FROM calibration_versions WHERE committed_at<=? '
        'ORDER BY calibration_id,revision',
        (knowledge_cutoff,),
    )
    latest = {}
    for row in rows:
        latest[row['calibration_id']] = dict(row)
    return latest


SAMPLE_COLUMNS = (
    'sample_id', 'aircraft', 'prediction_point', 'knowledge_cutoff', 'label_cutoff',
    'lookback_seconds', 'horizon_seconds',
    'buildable', 'label', 'label_state', 'features', 'lineage', 'reasons',
)


def canonical_sample(sample):
    """Normalized representation shared by digest, storage and reload."""
    return {
        'sample_id': sample['sample_id'],
        'aircraft': sample['aircraft'],
        'prediction_point': sample['prediction_point'],
        'knowledge_cutoff': sample['knowledge_cutoff'],
        'label_cutoff': sample['label_cutoff'],
        'lookback_seconds': sample['lookback_seconds'],
        'horizon_seconds': sample['horizon_seconds'],
        'buildable': bool(sample['buildable']),
        'label': None if sample.get('label') is None else bool(sample['label']),
        'label_state': sample['label_state'],
        'features': sample.get('features'),
        'lineage': sample['lineage'],
        'reasons': sample.get('reasons', []),
    }


def save_run(db, run_id, config, samples):
    identity(run_id, 'run_id')
    if not isinstance(samples, list) or not samples or len(samples) > 1_000_000:
        fail('samples_size', 'Training run requires a bounded nonempty sample list')
    normalized = [canonical_sample(sample) for sample in samples]
    checksum = digest({'config': config, 'samples': normalized})
    with db.transaction():
        if db.one('SELECT 1 FROM training_runs WHERE run_id=?', (run_id,)):
            fail('training_run_exists', 'Training run identifier is already used', run_id=run_id)
        db.execute(
            'INSERT INTO training_runs(run_id,config,digest,created_at) VALUES(?,?,?,?)',
            (run_id, canonical(config), checksum, db.clock()),
        )
        for sample in normalized:
            db.execute(
                'INSERT INTO training_samples(run_id,sample_id,aircraft,prediction_point,knowledge_cutoff,'
                'label_cutoff,lookback_seconds,horizon_seconds,buildable,label,label_state,features,lineage,reasons) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (
                    run_id, sample['sample_id'], sample['aircraft'], sample['prediction_point'],
                    sample['knowledge_cutoff'], sample['label_cutoff'],
                    sample['lookback_seconds'], sample['horizon_seconds'],
                    int(sample['buildable']),
                    None if sample['label'] is None else int(sample['label']),
                    sample['label_state'],
                    canonical(sample['features']) if sample['features'] is not None else None,
                    canonical(sample['lineage']),
                    canonical(sample['reasons']),
                ),
            )
        journal.record(db, 'training.run_saved', {'run_id': run_id, 'digest': checksum, 'samples': len(normalized)})
    return get_run(db, run_id)


def get_run(db, run_id):
    identity(run_id, 'run_id')
    row = db.one('SELECT * FROM training_runs WHERE run_id=?', (run_id,))
    if row is None:
        fail('unknown_training_run', 'Training run does not exist', run_id=run_id)
    row['config'] = json.loads(row['config'])
    row['samples'] = list_samples(db, run_id)
    recomputed = digest({'config': row['config'], 'samples': [canonical_sample(sample) for sample in row['samples']]})
    if recomputed != row['digest']:
        fail('training_run_corrupt', 'Training run digest mismatch')
    return row


def list_samples(db, run_id, *, buildable=None):
    clauses = 'run_id=?'
    args = [run_id]
    if buildable is not None:
        if type(buildable) is not bool:
            fail('invalid_boolean', 'buildable must be boolean')
        clauses += ' AND buildable=?'
        args.append(int(buildable))
    rows = db.all(
        f'SELECT * FROM training_samples WHERE {clauses} ORDER BY aircraft,prediction_point,sample_id',
        args,
    )
    for row in rows:
        row['buildable'] = bool(row['buildable'])
        row['label'] = None if row['label'] is None else bool(row['label'])
        row['features'] = json.loads(row['features']) if row['features'] is not None else None
        row['lineage'] = json.loads(row['lineage'])
        row['reasons'] = json.loads(row['reasons'])
    return rows


def save_split(db, run_id, split_id, config, manifest):
    identity(run_id, 'run_id')
    identity(split_id, 'split_id')
    get_run(db, run_id)
    checksum = digest(manifest)
    with db.transaction():
        if db.one('SELECT 1 FROM training_splits WHERE run_id=? AND split_id=?', (run_id, split_id)):
            fail('training_split_exists', 'Split identifier is already used', run_id=run_id, split_id=split_id)
        db.execute(
            'INSERT INTO training_splits(run_id,split_id,config,digest,manifest,created_at) VALUES(?,?,?,?,?,?)',
            (run_id, split_id, canonical(config), checksum, canonical(manifest), db.clock()),
        )
        journal.record(db, 'training.split_saved', {'run_id': run_id, 'split_id': split_id, 'digest': checksum})
    return get_split(db, run_id, split_id)


def get_split(db, run_id, split_id):
    row = db.one('SELECT * FROM training_splits WHERE run_id=? AND split_id=?', (run_id, split_id))
    if row is None:
        fail('unknown_training_split', 'Training split does not exist', run_id=run_id, split_id=split_id)
    row['config'] = json.loads(row['config'])
    row['manifest'] = json.loads(row['manifest'])
    if digest(row['manifest']) != row['digest']:
        fail('training_split_corrupt', 'Split manifest digest mismatch')
    return row


def list_splits(db, run_id):
    return [get_split(db, run_id, row['split_id']) for row in db.all(
        'SELECT split_id FROM training_splits WHERE run_id=? ORDER BY created_at,split_id', (run_id,))]


def list_runs(db):
    rows = db.all('SELECT run_id,digest,created_at FROM training_runs ORDER BY created_at,run_id')
    for row in rows:
        row['sample_count'] = db.scalar(
            'SELECT COUNT(*) FROM training_samples WHERE run_id=?', (row['run_id'],))
        row['buildable_count'] = db.scalar(
            'SELECT COUNT(*) FROM training_samples WHERE run_id=? AND buildable=1', (row['run_id'],))
    return rows

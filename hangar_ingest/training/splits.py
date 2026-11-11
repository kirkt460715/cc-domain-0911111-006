"""Grouped, time-forward data splits with an auditable manifest.

Leakage through splits is prevented by three hard rules:

1. Every sample of one aircraft belongs to exactly one fold. An aircraft can
   never appear in both train and test.
2. Folds are ordered in time: fold ``k`` is trained only on samples whose
   prediction point precedes every prediction point in fold ``k`` (expanding
   window). Within an aircraft the same ordering holds, so no fold trains on a
   later prediction to score an earlier one.
3. Samples whose label windows overlap are kept in the same fold. Two samples
   overlap when their ``[prediction_point, horizon_end)`` intervals intersect;
   connected overlap components are never separated.

The manifest records every excluded sample with its reason, every fold's
aircraft and time bounds, and a digest, so a split can be re-verified.
"""

from hangar_ingest.types import digest, fail, identity, integer, micros
from hangar_ingest.training import store

MIN_FOLDS = 2
MAX_FOLDS = 20


def _horizon_end(sample):
    return micros(sample['prediction_point']) + sample['horizon_seconds'] * 1_000_000


def _buildable_samples(run):
    samples = run['samples']
    buildable = [sample for sample in samples if sample['buildable']]
    excluded = [
        {
            'sample_id': sample['sample_id'],
            'aircraft': sample['aircraft'],
            'prediction_point': sample['prediction_point'],
            'reason': sample['label_state'],
            'detail': sample['reasons'],
        }
        for sample in samples
        if not sample['buildable']
    ]
    return buildable, excluded


def _overlap_components(samples):
    """Union-find over samples sharing an aircraft with overlapping horizons."""
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_aircraft = {}
    for index, sample in enumerate(samples):
        by_aircraft.setdefault(sample['aircraft'], []).append(index)
    for indices in by_aircraft.values():
        ordered = sorted(indices, key=lambda idx: (micros(samples[idx]['prediction_point']), idx))
        for position, index in enumerate(ordered):
            end = _horizon_end(samples[index])
            for other in ordered[position + 1:]:
                if micros(samples[other]['prediction_point']) >= end:
                    break
                union(index, other)
    groups = {}
    for index in range(len(samples)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _assign_folds(samples, fold_count):
    """Rank aircraft by their earliest prediction point and deal aircraft
    deterministically to folds so fold time bounds stay ordered."""
    aircraft = sorted({sample['aircraft'] for sample in samples})
    first_seen = {
        name: min(micros(sample['prediction_point']) for sample in samples if sample['aircraft'] == name)
        for name in aircraft
    }
    ordered = sorted(aircraft, key=lambda name: (first_seen[name], name))
    if len(ordered) < fold_count:
        fail(
            'fold_count',
            'Need at least as many aircraft as folds',
            aircraft=len(ordered),
            folds=fold_count,
        )
    assignment = {}
    # Spread earliest aircraft first while keeping each fold's time range
    # non-decreasing: fold k receives the k-th block in time order.
    bounds = [len(ordered) * k // fold_count for k in range(fold_count)] + [len(ordered)]
    for fold in range(fold_count):
        for name in ordered[bounds[fold]:bounds[fold + 1]]:
            assignment[name] = fold
    return assignment


def make_manifest(db, run_id, fold_count=3):
    identity(run_id, 'run_id')
    integer(fold_count, 'fold_count', MIN_FOLDS, MAX_FOLDS)
    run = store.get_run(db, run_id)
    samples, excluded = _buildable_samples(run)
    if not samples:
        fail('no_buildable_samples', 'Run has no buildable samples to split')
    components = _overlap_components(samples)
    assignment = _assign_folds(samples, fold_count)

    folds = [
        {
            'fold': fold,
            'sample_ids': [],
            'aircraft': set(),
            'min_prediction_point': None,
            'max_prediction_point': None,
            'max_horizon_end': None,
        }
        for fold in range(fold_count)
    ]
    component_audit = []
    for indices in components:
        folds_for_component = {assignment[samples[index]['aircraft']] for index in indices}
        if len(folds_for_component) > 1:
            # An overlap component must live in one fold. Anchor it to the
            # earliest fold any member belongs to and record the reassignment.
            anchor = min(folds_for_component)
            for index in indices:
                assignment[samples[index]['aircraft']] = anchor
            component_audit.append({
                'sample_ids': sorted(samples[index]['sample_id'] for index in indices),
                'folds_before': sorted(folds_for_component),
                'anchored_fold': anchor,
                'reason': 'overlapping_label_horizons',
            })
    for sample in samples:
        fold = assignment[sample['aircraft']]
        entry = folds[fold]
        entry['sample_ids'].append(sample['sample_id'])
        entry['aircraft'].add(sample['aircraft'])
        point = micros(sample['prediction_point'])
        end = _horizon_end(sample)
        entry['min_prediction_point'] = point if entry['min_prediction_point'] is None else min(entry['min_prediction_point'], point)
        entry['max_prediction_point'] = point if entry['max_prediction_point'] is None else max(entry['max_prediction_point'], point)
        entry['max_horizon_end'] = end if entry['max_horizon_end'] is None else max(entry['max_horizon_end'], end)

    serialized_folds = []
    from hangar_ingest.types import from_micros

    for entry in folds:
        serialized_folds.append({
            'fold': entry['fold'],
            'sample_ids': sorted(entry['sample_ids']),
            'aircraft': sorted(entry['aircraft']),
            'min_prediction_point': from_micros(entry['min_prediction_point']) if entry['min_prediction_point'] is not None else None,
            'max_prediction_point': from_micros(entry['max_prediction_point']) if entry['max_prediction_point'] is not None else None,
            'max_horizon_end': from_micros(entry['max_horizon_end']) if entry['max_horizon_end'] is not None else None,
            'positive': sum(1 for sid in entry['sample_ids'] if _label_of(run, sid) is True),
            'negative': sum(1 for sid in entry['sample_ids'] if _label_of(run, sid) is False),
        })
    _assert_no_shared_aircraft(serialized_folds)
    _assert_time_order(serialized_folds)
    manifest = {
        'schema': 'hangar-training-split',
        'version': 1,
        'run_id': run_id,
        'fold_count': fold_count,
        'rule': 'grouped_aircraft_expanding_window_overlap_anchored',
        'folds': serialized_folds,
        'excluded_samples': excluded,
        'overlap_reassignments': component_audit,
        'sample_count': len(samples),
    }
    manifest['digest'] = digest({key: value for key, value in manifest.items() if key != 'digest'})
    return manifest


def _label_of(run, sample_id):
    for sample in run['samples']:
        if sample['sample_id'] == sample_id:
            return sample['label']
    return None  # pragma: no cover - ids come from the same run


def _assert_no_shared_aircraft(folds):
    seen = {}
    for fold in folds:
        for aircraft in fold['aircraft']:
            if aircraft in seen:
                fail('split_shared_aircraft', 'Aircraft appears in multiple folds', aircraft=aircraft, folds=[seen[aircraft], fold['fold']])
            seen[aircraft] = fold['fold']


def _assert_time_order(folds):
    previous_max = None
    for fold in folds:
        if not fold['sample_ids']:
            continue
        if previous_max is not None and fold['min_prediction_point'] < previous_max:
            fail(
                'split_time_order',
                'Fold prediction windows are not time-ordered',
                fold=fold['fold'],
                previous_max=previous_max,
                min_point=fold['min_prediction_point'],
            )
        previous_max = fold['max_prediction_point']


def verify_manifest(db, run_id, split_id):
    stored = store.get_split(db, run_id, split_id)
    recomputed = make_manifest(db, run_id, stored['manifest']['fold_count'])
    ok = recomputed['digest'] == stored['manifest']['digest']
    return {
        'ok': ok,
        'run_id': run_id,
        'split_id': split_id,
        'stored_digest': stored['manifest']['digest'],
        'recomputed_digest': recomputed['digest'],
    }


def save_manifest(db, run_id, split_id, fold_count=3):
    manifest = make_manifest(db, run_id, fold_count)
    return store.save_split(db, run_id, split_id, {'fold_count': fold_count}, manifest)

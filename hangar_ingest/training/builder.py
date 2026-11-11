"""Orchestrate leakage-safe training sample construction.

A candidate is the tuple (aircraft, prediction point, lookback window,
prediction span, label condition). Feature and label construction run against
separate frozen cutoffs. Any failure to reconstruct historical evidence marks
the candidate ``buildable=false`` with a machine-readable reason; a failed
candidate is never silently dropped and never turned into a negative.
"""

from hangar_ingest.types import PipelineError, digest, fail, identity, instant, integer, micros
from hangar_ingest.training import features as features_module
from hangar_ingest.training import labels as labels_module
from hangar_ingest.training import point_in_time as pit

CONFIG_FIELDS = {
    'aircraft', 'prediction_points', 'lookback_seconds', 'horizon_seconds',
    'knowledge_cutoff', 'label_cutoff', 'label_condition',
}
CONFIG_OPTIONAL = {'quantiles', 'tolerance_seconds', 'outlier_multiplier', 'components'}


def validate_config(config):
    if not isinstance(config, dict):
        fail('sample_config', 'Training configuration must be an object')
    unknown = set(config) - CONFIG_FIELDS - CONFIG_OPTIONAL
    if unknown:
        fail('sample_config', 'Unknown training configuration fields', unknown=sorted(unknown))
    missing = CONFIG_FIELDS - set(config)
    if missing:
        fail('sample_config', 'Missing training configuration fields', missing=sorted(missing))
    identity(config['aircraft'], 'aircraft')
    integer(config['lookback_seconds'], 'lookback_seconds', 1, 366 * 86400)
    integer(config['horizon_seconds'], 'horizon_seconds', 1, 366 * 86400)
    knowledge_cutoff = instant(config['knowledge_cutoff'], 'knowledge_cutoff')
    label_cutoff = instant(config['label_cutoff'], 'label_cutoff')
    if label_cutoff < knowledge_cutoff:
        fail('cutoff_order', 'Label observation cutoff must be at or after the knowledge cutoff')
    points = config['prediction_points']
    if not isinstance(points, list) or not points or len(points) > 100000:
        fail('prediction_points', 'Need a bounded nonempty prediction point list')
    normalized_points = []
    for value in points:
        point = instant(value, 'prediction_point')
        if point > knowledge_cutoff:
            fail('temporal_order', 'Prediction point must not be after the knowledge cutoff')
        normalized_points.append(point)
    if len(set(normalized_points)) != len(normalized_points):
        fail('prediction_points', 'Prediction points must be distinct')
    condition = labels_module.validate_condition(config['label_condition'])
    result = {
        'aircraft': config['aircraft'],
        'prediction_points': normalized_points,
        'lookback_seconds': config['lookback_seconds'],
        'horizon_seconds': config['horizon_seconds'],
        'knowledge_cutoff': knowledge_cutoff,
        'label_cutoff': label_cutoff,
        'label_condition': condition,
        'quantiles': config.get('quantiles', ('0.05', '0.5', '0.95')),
        'tolerance_seconds': config.get('tolerance_seconds', 300),
        'outlier_multiplier': config.get('outlier_multiplier', '3'),
        'components': config.get('components'),
    }
    features_module._quantile_list(list(result['quantiles']))
    integer(result['tolerance_seconds'], 'tolerance_seconds', 1, 86400)
    return result


def _window_start(prediction_point, lookback_seconds):
    from hangar_ingest.types import from_micros

    return from_micros(micros(prediction_point) - lookback_seconds * 1_000_000)


def _sample_id(config, point):
    return digest({
        'aircraft': config['aircraft'],
        'prediction_point': point,
        'lookback_seconds': config['lookback_seconds'],
        'horizon_seconds': config['horizon_seconds'],
        'knowledge_cutoff': config['knowledge_cutoff'],
        'label_cutoff': config['label_cutoff'],
        'label_condition': config['label_condition'],
    })[:24]


def build_sample(db, config, point):
    window_start = _window_start(point, config['lookback_seconds'])
    sample = {
        'sample_id': _sample_id(config, point),
        'aircraft': config['aircraft'],
        'prediction_point': point,
        'knowledge_cutoff': config['knowledge_cutoff'],
        'label_cutoff': config['label_cutoff'],
        'window_start': window_start,
        'lookback_seconds': config['lookback_seconds'],
        'horizon_seconds': config['horizon_seconds'],
        'buildable': False,
        'label': None,
        'label_state': 'unbuilt',
        'features': None,
        'lineage': {},
        'reasons': [],
    }
    try:
        window = features_module.build_window(
            db, config['aircraft'], window_start, point, config['knowledge_cutoff'],
            lookback_seconds=config['lookback_seconds'], horizon_seconds=config['horizon_seconds'],
            quantiles=config['quantiles'], tolerance_seconds=config['tolerance_seconds'],
            outlier_multiplier=config['outlier_multiplier'], components=config['components'],
        )
        grade = labels_module.resolve_label(
            db, config['aircraft'], point, config['horizon_seconds'],
            config['label_cutoff'], config['label_condition'],
        )
    except PipelineError as exc:
        sample['reasons'] = [exc.as_dict()]
        sample['label_state'] = 'unbuildable'
        sample['lineage'] = _cutoff_lineage(db, config, point)
        return sample
    sample['features'] = window
    sample['label_state'] = grade['state']
    sample['label'] = grade['label']
    sample['lineage'] = {
        'config_digest': digest(config),
        'cutoffs': {
            'knowledge_cutoff': config['knowledge_cutoff'],
            'label_cutoff': config['label_cutoff'],
        },
        'label_evidence': grade['evidence'],
        'label_trigger': grade.get('trigger'),
        'known_orders_at_knowledge_cutoff': sorted(
            pit.work_orders_known_at(db, config['knowledge_cutoff'])),
        'device_lineage': [block['lineage'] for block in window['devices']],
        'source_revisions': _source_revisions(window, grade),
    }
    sample['buildable'] = grade['state'] in {'positive', 'negative'}
    if not sample['buildable']:
        sample['reasons'] = [{'code': 'right_censored', 'message': grade['reason'], 'context': {
            'horizon_end': grade['horizon_end'], 'label_cutoff': grade['label_cutoff']}}]
    return sample


def _cutoff_lineage(db, config, point):
    try:
        orders = pit.work_orders_known_at(db, config['knowledge_cutoff'])
        calibrations = pit.calibrations_known_at(db, config['knowledge_cutoff'])
    except PipelineError as exc:  # pragma: no cover - defensive, failure reason already captured
        return {'lineage_error': exc.code}
    return {
        'config_digest': digest(config),
        'known_orders_at_knowledge_cutoff': sorted(orders),
        'known_calibrations_at_knowledge_cutoff': sorted(calibrations),
    }


def _source_revisions(window, grade):
    revisions = set()
    for block in window['devices']:
        for item in block['lineage']['calibration_parameters']:
            revisions.add(f"calibration:{item['calibration_id']}@{item['revision']}")
        for order_id in block['lineage'].get('maintenance_orders', []):
            revisions.add(f"order:{order_id}")
    for item in grade['evidence']:
        revisions.add(f"order:{item['order_id']}@{item['revision']}")
    return sorted(revisions)


def build_multi_aircraft_run(db, config):
    """Build one run spanning several aircraft for grouped splitting.

    The configuration carries an ``aircrafts`` list instead of ``aircraft``;
    every other parameter (cutoffs, windows, label condition) is shared, which
    keeps cross-aircraft samples directly comparable inside one run.
    """
    if not isinstance(config, dict) or 'aircrafts' not in config:
        fail('sample_config', 'Multi-aircraft run requires an aircrafts list')
    aircrafts = config['aircrafts']
    if not isinstance(aircrafts, list) or not aircrafts or len(aircrafts) > 1000:
        fail('sample_config', 'aircrafts must be a bounded nonempty list')
    if len(set(aircrafts)) != len(aircrafts):
        fail('sample_config', 'aircrafts must be distinct')
    points_by_aircraft = config.get('prediction_points_by_aircraft')
    shared_points = config.get('prediction_points')
    if points_by_aircraft is not None:
        if not isinstance(points_by_aircraft, dict) or set(points_by_aircraft) != set(aircrafts):
            fail('sample_config', 'prediction_points_by_aircraft must cover exactly the aircrafts list')
    elif shared_points is None:
        fail('sample_config', 'Multi-aircraft run requires prediction points')
    all_samples = []
    normalized_configs = []
    for aircraft in aircrafts:
        per_aircraft = {key: value for key, value in config.items()
                        if key not in ('aircrafts', 'prediction_points_by_aircraft')}
        per_aircraft['aircraft'] = identity(aircraft, 'aircraft')
        if points_by_aircraft is not None:
            per_aircraft['prediction_points'] = points_by_aircraft[aircraft]
        normalized = validate_config(per_aircraft)
        normalized_configs.append(normalized)
        all_samples.extend(build_sample(db, normalized, point) for point in normalized['prediction_points'])
    combined_config = {
        'aircrafts': aircrafts,
        'shared': {key: normalized_configs[0][key] for key in (
            'lookback_seconds', 'horizon_seconds', 'knowledge_cutoff', 'label_cutoff',
            'label_condition', 'quantiles', 'tolerance_seconds', 'outlier_multiplier', 'components')},
        'prediction_points_by_aircraft': {
            normalized['aircraft']: normalized['prediction_points'] for normalized in normalized_configs
        },
    }
    return {
        'config': combined_config,
        'samples': all_samples,
        'summary': _summarize(all_samples),
    }


def save_multi_aircraft_run(db, run_id, config):
    from hangar_ingest.training import store

    result = build_multi_aircraft_run(db, config)
    stored = store.save_run(db, run_id, result['config'], result['samples'])
    return {'run': stored, 'summary': result['summary']}


def _summarize(samples):
    return {
        'aircraft': sorted({sample['aircraft'] for sample in samples}),
        'candidates': len(samples),
        'buildable': sum(1 for sample in samples if sample['buildable']),
        'right_censored': sum(1 for sample in samples if sample['label_state'] == 'censored'),
        'unbuildable': sum(1 for sample in samples if sample['label_state'] == 'unbuildable'),
        'positive': sum(1 for sample in samples if sample['label'] is True),
        'negative': sum(1 for sample in samples if sample['label'] is False),
        'failure_codes': _failure_codes(samples),
    }


def build_run(db, config):
    config = validate_config(config)
    samples = [build_sample(db, config, point) for point in config['prediction_points']]
    summary = _summarize(samples)
    summary['aircraft'] = config['aircraft']
    return {'config': config, 'samples': samples, 'summary': summary}


def _failure_codes(samples):
    counts = {}
    for sample in samples:
        for reason in sample['reasons']:
            counts[reason['code']] = counts.get(reason['code'], 0) + 1
    return dict(sorted(counts.items()))


def save_run(db, run_id, config):
    from hangar_ingest.training import store

    result = build_run(db, config)
    stored = store.save_run(db, run_id, result['config'], result['samples'])
    return {'run': stored, 'summary': result['summary']}

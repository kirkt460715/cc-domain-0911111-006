"""Feature window construction with full point-in-time provenance.

Every numeric feature is computed only from observations that satisfy both
clocks in :mod:`hangar_ingest.training.point_in_time`: the event happened before
the prediction point and the row was committed by the knowledge cutoff.

Calibration values are taken from the frozen revision visible at the cutoff,
not from the mutable current calibration row, so post-hoc corrections cannot
change historical features. Each device block lists the source observation and
calibration revisions needed to reproduce the numbers.
"""

from decimal import Decimal, localcontext

from hangar_ingest.quality.coverage import complement, merge_intervals
from hangar_ingest.quality.profiles import percentile
from hangar_ingest.types import decimal_string, fail, from_micros, instant, integer, micros
from hangar_ingest.training import point_in_time as pit


def _quantile_list(quantiles):
    if not isinstance(quantiles, list) or not quantiles or len(quantiles) > 20:
        fail('quantile_config', 'Need a bounded nonempty quantile list')
    result = []
    for value in quantiles:
        q = Decimal(str(value))
        if not 0 <= q <= 1:
            fail('quantile_range', 'Quantile must be between zero and one')
        result.append(value)
    if len(set(Decimal(str(v)) for v in result)) != len(result):
        fail('quantile_config', 'Quantiles must be distinct')
    return result


def _iqr_outlier_ratio(values, multiplier):
    if len(values) < 4:
        return None, 'fewer_than_four_samples'
    q1 = Decimal(percentile(values, '0.25'))
    q3 = Decimal(percentile(values, '0.75'))
    spread = q3 - q1
    lower = q1 - multiplier * spread
    upper = q3 + multiplier * spread
    flagged = sum(1 for value in values if value < lower or value > upper)
    with localcontext() as context:
        context.prec = 28
        return decimal_string(Decimal(flagged) / Decimal(len(values))), 'iqr'


def _coverage_ratio(rows, start_us, end_us, tolerance_seconds):
    half = tolerance_seconds * 1_000_000 // 2
    intervals = [
        [max(start_us, micros(row['event_time']) - half), min(end_us, micros(row['event_time']) + half)]
        for row in rows
    ]
    covered = merge_intervals(intervals)
    covered_us = sum(right - left for left, right in covered)
    total_us = end_us - start_us
    missing = complement(covered, start_us, end_us)
    return {
        'coverage_ratio': decimal_string(Decimal(covered_us) / Decimal(total_us)),
        'covered_us': covered_us,
        'total_us': total_us,
        'missing_intervals': [
            {'start': from_micros(left), 'end': from_micros(right), 'duration_us': right - left}
            for left, right in missing
        ],
    }


def _nearest_maintenance(orders, device, prediction_point):
    """Distance to the closest already-known closed maintenance event.

    Only orders visible at the knowledge cutoff and closed on or before the
    prediction point qualify. An order that was still open, unknown, or
    cancelled at the cutoff is ignored even if it exists in current state.
    """
    point_us = micros(prediction_point)
    candidates = pit.matching_orders(
        orders, device['aircraft'], device['component'],
        device.get('_horizon_start', prediction_point), prediction_point,
    )
    usable = [order for order in candidates if order['state'] == 'closed' and micros(order['closed_at']) <= point_us]
    if not usable:
        return {'distance_seconds': None, 'order_id': None, 'closed_at': None, 'known_revision': None}
    nearest = min(usable, key=lambda order: (point_us - micros(order['closed_at']), order['order_id']))
    return {
        'distance_seconds': (point_us - micros(nearest['closed_at'])) // 1_000_000,
        'order_id': nearest['order_id'],
        'closed_at': nearest['closed_at'],
        'known_revision': nearest['revision'],
    }


def build_device_features(db, device, window_start, prediction_point, knowledge_cutoff, *,
                          quantiles, tolerance_seconds, outlier_multiplier, orders, calibrations):
    start = instant(window_start)
    point = instant(prediction_point)
    cutoff = instant(knowledge_cutoff, 'knowledge_cutoff')
    rows = pit.eligible_observations(db, device['device_id'], start, point, cutoff)
    values = [Decimal(row['value']) for row in rows]
    count = len(values)
    block = {
        'device_id': device['device_id'],
        'metric': device['metric'],
        'unit': device['canonical_unit'],
        'window_start': start,
        'window_end': point,
        'knowledge_cutoff': cutoff,
        'samples': count,
        'mean': None,
        'quantiles': {str(q): None for q in quantiles},
        'coverage': _coverage_ratio(rows, micros(start), micros(point), tolerance_seconds),
        'outlier_ratio': None,
        'outlier_method': 'iqr' if count >= 4 else 'fewer_than_four_samples',
        'nearest_maintenance': None,
        'lineage': {
            'observation_ids': [row['observation_id'] for row in rows],
            'observation_commits': {row['observation_id']: row['created_at'] for row in rows},
            'calibrations': sorted({
                row['calibration_id'] for row in rows if row['calibration_id'] is not None
            }),
            'calibration_revisions': sorted({
                f"{row['calibration_id']}@{row['calibration_revision']}"
                for row in rows
                if row['calibration_id'] is not None
            }),
            'maintenance_orders': [],
        },
    }
    if values:
        with localcontext() as context:
            context.prec = 28
            block['mean'] = decimal_string(sum(values, Decimal(0)) / count)
        for quantile in quantiles:
            block['quantiles'][str(quantile)] = percentile(values, str(quantile))
        ratio, method = _iqr_outlier_ratio(values, Decimal(str(outlier_multiplier)))
        block['outlier_ratio'] = ratio
        block['outlier_method'] = method
    _assert_calibration_evidence(rows, calibrations)
    block['lineage']['calibration_parameters'] = [
        {
            'calibration_id': row['calibration_id'],
            'revision': row['calibration_revision'],
            'scale': calibrations[row['calibration_id']]['scale'],
            'offset': calibrations[row['calibration_id']]['offset'],
            'clock_offset_us': calibrations[row['calibration_id']]['clock_offset_us'],
            'committed_at': calibrations[row['calibration_id']]['committed_at'],
        }
        for row in rows
        if row['calibration_id'] is not None
    ]
    device_view = dict(device)
    device_view['_horizon_start'] = start
    block['nearest_maintenance'] = _nearest_maintenance(orders, device_view, point)
    if block['nearest_maintenance']['order_id'] is not None:
        block['lineage']['maintenance_orders'].append(block['nearest_maintenance']['order_id'])
    return block


def _assert_calibration_evidence(rows, calibrations):
    """Every calibration referenced by a used row must have a frozen parameter
    revision visible at the cutoff, matching the revision used at ingest.

    Missing historical evidence makes the sample fail closed instead of
    silently reading the mutable current calibration row."""
    for row in rows:
        if row['calibration_id'] is None:
            continue
        frozen = calibrations.get(row['calibration_id'])
        if frozen is None:
            fail(
                'historical_evidence_missing',
                'Feature window references a calibration unknown at the cutoff',
                calibration_id=row['calibration_id'],
                observation_id=row['observation_id'],
            )
        if frozen['revision'] != row['calibration_revision']:
            fail(
                'calibration_revision_drift',
                'Cutoff-visible calibration revision differs from the revision used at ingest',
                calibration_id=row['calibration_id'],
                stored_revision=row['calibration_revision'],
                known_revision=frozen['revision'],
            )


def build_window(db, aircraft, window_start, prediction_point, knowledge_cutoff, *,
                 lookback_seconds, horizon_seconds, quantiles=('0.5', '0.9'),
                 tolerance_seconds=300, outlier_multiplier='3', components=None):
    integer(lookback_seconds, 'lookback_seconds', 1, 366 * 86400)
    integer(horizon_seconds, 'horizon_seconds', 0, 366 * 86400)
    integer(tolerance_seconds, 'tolerance_seconds', 1, 86400)
    quantiles = _quantile_list(list(quantiles))
    multiplier = Decimal(str(outlier_multiplier))
    if multiplier <= 0:
        fail('outlier_multiplier', 'Outlier multiplier must be positive')
    point = instant(prediction_point)
    cutoff = instant(knowledge_cutoff, 'knowledge_cutoff')
    orders = pit.work_orders_known_at(db, cutoff)
    calibrations = pit.calibrations_known_at(db, cutoff)
    devices = db.all('SELECT * FROM devices WHERE aircraft=? ORDER BY component,device_id', (aircraft,))
    if components is not None:
        if not isinstance(components, list) or not components:
            fail('component_config', 'Components filter must be a nonempty list')
        devices = [device for device in devices if device['component'] in components]
    if not devices:
        fail('unknown_aircraft', 'Aircraft has no registered devices in scope', aircraft=aircraft)
    blocks = [
        build_device_features(
            db, device, window_start, point, cutoff,
            quantiles=quantiles, tolerance_seconds=tolerance_seconds,
            outlier_multiplier=multiplier, orders=orders, calibrations=calibrations,
        )
        for device in devices
    ]
    return {
        'aircraft': aircraft,
        'prediction_point': point,
        'knowledge_cutoff': cutoff,
        'window_start': instant(window_start),
        'lookback_seconds': lookback_seconds,
        'horizon_seconds': horizon_seconds,
        'devices': blocks,
        'known_orders': sorted(orders),
    }

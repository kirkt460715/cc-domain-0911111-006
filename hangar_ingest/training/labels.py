"""Frozen label windows with explicit right-censoring.

Labels use a separate observation cutoff (``label_cutoff``) that is always at
or after the knowledge cutoff used for features. The two cutoffs answer
different questions:

* knowledge cutoff: what had the system been told by prediction time?
* label cutoff: how far into the future have outcomes been frozen for grading?

A sample whose horizon has not fully elapsed under the label cutoff is
*censored* and must never be treated as a negative, even when no matching work
order is currently visible.
"""

from hangar_ingest.types import fail, instant, integer, micros
from hangar_ingest.training import point_in_time as pit

VALID_STATES = {'open', 'closed', 'cancelled'}


def validate_condition(condition):
    if not isinstance(condition, dict):
        fail('label_condition', 'Label condition must be an object')
    allowed = {'components', 'issues', 'states', 'min_resolution_length'}
    unknown = set(condition) - allowed
    if unknown:
        fail('label_condition', 'Unknown label condition fields', unknown=sorted(unknown))
    components = condition.get('components')
    if components is not None and (not isinstance(components, list) or not components):
        fail('label_condition', 'Components filter must be a nonempty list')
    issues = condition.get('issues')
    if issues is not None and (not isinstance(issues, list) or not all(isinstance(value, str) and value for value in issues)):
        fail('label_condition', 'Issue filters must be nonempty strings')
    states = condition.get('states', ['closed'])
    if not isinstance(states, list) or not states or any(value not in VALID_STATES for value in states):
        fail('label_condition', 'States must be a nonempty subset of open/closed/cancelled')
    minimum = condition.get('min_resolution_length', 0)
    integer(minimum, 'min_resolution_length', 0, 366 * 86400)
    return {
        'components': components,
        'issues': issues,
        'states': states,
        'min_resolution_length': minimum,
    }


def _order_matches(order, aircraft, condition):
    if order['aircraft'] != aircraft or order['state'] not in condition['states']:
        return False
    if condition['components'] is not None and order['component'] not in condition['components']:
        return False
    if condition['issues'] is not None and not any(token in order['issue'] for token in condition['issues']):
        return False
    duration = (micros(order['closed_at']) - micros(order['opened_at'])) // 1_000_000
    return duration >= condition['min_resolution_length']


def resolve_label(db, aircraft, prediction_point, horizon_seconds, label_cutoff, condition):
    """Grade one candidate against the frozen follow-up window.

    Returns a state in ``positive`` / ``negative`` / ``censored`` plus the
    full evidence list. Only work order revisions committed by the label
    cutoff are considered; an order that appears later cannot create a label
    retroactively, and an order rewritten after the cutoff is read as-of it.
    """
    point = instant(prediction_point)
    cutoff = instant(label_cutoff, 'label_cutoff')
    integer(horizon_seconds, 'horizon_seconds', 1, 366 * 86400)
    condition = validate_condition(condition)
    horizon_end_us = micros(point) + horizon_seconds * 1_000_000
    from hangar_ingest.types import from_micros

    horizon_end = from_micros(horizon_end_us)
    if cutoff < point:
        fail('temporal_order', 'Label cutoff cannot precede the prediction point')
    orders = pit.work_orders_known_at(db, cutoff)
    if condition['components'] is None:
        components = sorted({order['component'] for order in orders.values() if order['aircraft'] == aircraft})
    else:
        components = condition['components']
    overlapping = []
    seen = set()
    for component in components:
        for order in pit.matching_orders(orders, aircraft, component, point, horizon_end):
            if order['order_id'] not in seen:
                seen.add(order['order_id'])
                overlapping.append(order)
    evidence = []
    for order in overlapping:
        evidence.append({
            'order_id': order['order_id'],
            'revision': order['revision'],
            'component': order['component'],
            'state': order['state'],
            'opened_at': order['opened_at'],
            'closed_at': order['closed_at'],
            'known_at': order['known_at'],
            'matches_condition': _order_matches(order, aircraft, condition),
        })
    if micros(cutoff) < horizon_end_us:
        return {
            'state': 'censored',
            'label': None,
            'reason': 'horizon_not_observed',
            'horizon_end': horizon_end,
            'label_cutoff': cutoff,
            'evidence': evidence,
        }
    positives = [item for item in evidence if item['matches_condition']]
    if positives:
        first = min(positives, key=lambda item: (micros(item['opened_at']), item['order_id']))
        return {
            'state': 'positive',
            'label': True,
            'reason': 'matching_work_order',
            'horizon_end': horizon_end,
            'label_cutoff': cutoff,
            'evidence': evidence,
            'trigger': {'order_id': first['order_id'], 'revision': first['revision'], 'opened_at': first['opened_at']},
        }
    return {
        'state': 'negative',
        'label': False,
        'reason': 'no_matching_work_order_in_frozen_horizon',
        'horizon_end': horizon_end,
        'label_cutoff': cutoff,
        'evidence': evidence,
    }

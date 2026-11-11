"""Rebuild what the system actually knew at a fixed knowledge cutoff.

Two clocks are kept separate on purpose:

* ``event_time`` says when something physically happened on the aircraft.
* the commit clock (``created_at`` / audit ``at`` / ``committed_at``) says when
  the system learned about it.

A feature may only use rows whose event happened before the prediction point
*and* whose commit was already visible at the knowledge cutoff. A work order or
calibration correction that landed after the cutoff cannot travel backwards.
"""

import json

from hangar_ingest.types import fail, instant, integer


def _audit_page(db, action, limit=100000):
    rows = db.all(
        'SELECT position,details,at FROM audit WHERE action=? ORDER BY position',
        (action,),
    )
    if len(rows) > limit:  # pragma: no cover - defensive bound, audit is append only
        fail('audit_scan', 'Audit history exceeds supported scan bound', action=action)
    for row in rows:
        row['details'] = json.loads(row['details'])
    return rows


def calibrations_known_at(db, knowledge_cutoff):
    """Calibration parameters exactly as committed at the cutoff.

    Sourced exclusively from the append-only ``calibration_versions`` table:
    revisions committed after the cutoff are invisible, so a later
    recalibration can never alter values used by a historical feature window.
    """
    from hangar_ingest.training.store import calibration_versions_known_at

    cutoff = instant(knowledge_cutoff, 'knowledge_cutoff')
    return calibration_versions_known_at(db, cutoff)


def calibration_for_device(calibrations, device_id, raw_time):
    raw_time = instant(raw_time)
    matches = [
        entry
        for entry in calibrations.values()
        if entry['device_id'] == device_id and entry['valid_from'] <= raw_time < entry['valid_until']
    ]
    if len(matches) > 1:
        fail(
            'calibration_overlap',
            'More than one known calibration covers this point',
            device_id=device_id,
            raw_time=raw_time,
        )
    return matches[0] if matches else None


def work_orders_known_at(db, knowledge_cutoff):
    """Work order revisions whose commit was visible at the cutoff.

    Each order is reduced to the highest revision committed on or before the
    cutoff. Orders whose first revision arrived later are absent, and a later
    revision that rewrote an interval or state cannot leak into the past.
    """
    cutoff = instant(knowledge_cutoff, 'knowledge_cutoff')
    latest = {}
    for row in _audit_page(db, 'order.updated'):
        if row['at'] > cutoff:
            continue
        order_id = row['details']['order_id']
        revision = integer(row['details']['revision'], 'revision', 1)
        if order_id not in latest or revision > latest[order_id]['revision']:
            payload = db.one(
                'SELECT payload FROM order_versions WHERE order_id=? AND revision=?',
                (order_id, revision),
            )
            if payload is None:  # pragma: no cover - version rows are written with the audit entry
                fail('historical_evidence_missing', 'Work order revision payload is missing', order_id=order_id, revision=revision)
            latest[order_id] = {**json.loads(payload['payload']), 'known_at': row['at']}
    return dict(sorted(latest.items()))


def order_as_of(db, order_id, knowledge_cutoff):
    return work_orders_known_at(db, knowledge_cutoff).get(order_id)


def eligible_observations(db, device_id, window_start, prediction_point, knowledge_cutoff):
    """Rows that had both happened and been committed before use.

    Both predicates are mandatory: an old event ingested after the cutoff is
    still invisible, even though ``event_time`` lies inside the feature window.
    """
    start = instant(window_start)
    point = instant(prediction_point)
    cutoff = instant(knowledge_cutoff, 'knowledge_cutoff')
    if not start < point <= cutoff:
        fail('temporal_order', 'Need window start < prediction point <= knowledge cutoff')
    return db.all(
        'SELECT * FROM observations WHERE device_id=? AND event_time>=? AND event_time<? '
        'AND created_at<=? ORDER BY event_time,observation_id',
        (device_id, start, point, cutoff),
    )


def matching_orders(orders, aircraft, component, start, end):
    start = instant(start)
    end = instant(end)
    return [
        order
        for order in orders.values()
        if order['aircraft'] == aircraft
        and order['component'] == component
        and order['state'] != 'cancelled'
        and order['opened_at'] < end
        and order['closed_at'] > start
    ]

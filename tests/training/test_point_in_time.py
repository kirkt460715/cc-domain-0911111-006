import datetime as dt
from decimal import Decimal

from hangar_ingest.types import PipelineError, digest
from hangar_ingest.quality import devices, rules
from hangar_ingest.decoders import workorders
from hangar_ingest.runs import processor
from hangar_ingest.training import builder, features, labels, point_in_time as pit, splits, store
from tests.support import PipelineCase, sample, order


def iso(**delta):
    base = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
    return (base + dt.timedelta(**delta)).isoformat(timespec='microseconds').replace('+00:00', 'Z')


class Clock:
    def __init__(self):
        self.point = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)

    def __call__(self):
        return self.point.isoformat(timespec='microseconds').replace('+00:00', 'Z')

    def move(self, when):
        self.point = dt.datetime.fromisoformat(when.replace('Z', '+00:00'))


class KnowledgeCutoffCase(PipelineCase):
    def setUp(self):
        # Build the same fixtures as PipelineCase but with a controllable commit
        # clock starting on Nov 1, before every knowledge cutoff used below.
        import tempfile
        from pathlib import Path
        from hangar_ingest.runs.database import Database

        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'pipeline.db'
        self.clock = Clock()
        self.db = Database(self.path, clock=self.clock)
        devices.register(self.db, 'dev1', 'plane1', 'engine1', 'temperature', 'C')
        devices.add_alias(self.db, 'feed1', 'sensor1', 'dev1', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        devices.add_calibration(self.db, 'cal1', 'dev1', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        rules.configure(self.db, 'dev1', minimum='-100', maximum='200')
        devices.register(self.db, 'dev2', 'plane2', 'engine1', 'temperature', 'C')
        devices.add_alias(self.db, 'feed1', 'sensor2', 'dev2', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        devices.add_calibration(self.db, 'cal2', 'dev2', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        rules.configure(self.db, 'dev2', minimum='-100', maximum='200')

    def ingest_at(self, rows, when, batch_id):
        self.clock.move(when)
        self.ingest(rows, batch_id=batch_id)

    def order_at(self, payload, when):
        self.clock.move(when)
        return workorders.upsert(self.db, payload)


class PointInTimeTests(KnowledgeCutoffCase):
    def test_late_ingested_old_event_is_invisible_at_cutoff(self):
        # Event physically occurs on Nov 1 but only arrives Nov 5.
        self.ingest_at([sample(1, '20', time=iso(days=0, hours=10))], iso(days=5), 'late')
        cutoff = iso(days=4)
        rows = pit.eligible_observations(self.db, 'dev1', iso(days=0), iso(days=3), cutoff)
        self.assertEqual(rows, [])
        rows_after = pit.eligible_observations(self.db, 'dev1', iso(days=0), iso(days=6), iso(days=6))
        self.assertEqual(len(rows_after), 1)

    def test_work_order_revision_after_cutoff_does_not_travel_back(self):
        self.order_at(order(opened_at=iso(days=1), closed_at=iso(days=2), issue='vibration'), iso(days=3))
        cutoff = iso(days=4)
        known = pit.work_orders_known_at(self.db, cutoff)
        self.assertEqual(known['order1']['issue'], 'vibration')
        # Correction lands Nov 8 with a rewritten issue and interval.
        self.order_at(order(revision=2, issue='retroactively rewritten fuel leak',
                            opened_at=iso(days=1), closed_at=iso(days=2, hours=6)), iso(days=8))
        known_old = pit.work_orders_known_at(self.db, cutoff)
        self.assertEqual(known_old['order1']['issue'], 'vibration')
        self.assertEqual(known_old['order1']['revision'], 1)
        known_new = pit.work_orders_known_at(self.db, iso(days=9))
        self.assertEqual(known_new['order1']['revision'], 2)

    def test_calibration_correction_is_frozen_per_revision(self):
        self.ingest_at([sample(1, '10', time=iso(days=0, hours=2))], iso(days=0, hours=3), 'b1')
        self.clock.point = dt.datetime.fromisoformat(iso(days=4).replace('Z', '+00:00'))
        devices.correct_calibration(self.db, 'cal1', 1, '2', '0', 0)
        versions = store.calibration_versions_known_at(self.db, iso(days=3))
        self.assertEqual(versions['cal1']['revision'], 1)
        self.assertEqual(versions['cal1']['scale'], '1')
        versions_after = store.calibration_versions_known_at(self.db, iso(days=5))
        self.assertEqual(versions_after['cal1']['revision'], 2)
        self.assertEqual(versions_after['cal1']['scale'], '2')


class FeatureWindowTests(KnowledgeCutoffCase):
    def config(self, point=iso(days=3), cutoff=iso(days=4), **changes):
        config = {
            'aircraft': 'plane1',
            'prediction_points': [point],
            'lookback_seconds': 3 * 86400,
            'horizon_seconds': 2 * 86400,
            'knowledge_cutoff': cutoff,
            'label_cutoff': iso(days=10),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
            'quantiles': ['0.5', '0.9'],
            'tolerance_seconds': 3600,
        }
        config.update(changes)
        return config

    def test_window_statistics_coverage_outliers_and_maintenance_gap(self):
        rows = [
            sample(1, '10', time=iso(days=1, hours=0)),
            sample(2, '12', time=iso(days=1, hours=1)),
            sample(3, '11', time=iso(days=1, hours=2)),
            sample(4, '99', time=iso(days=1, hours=3)),
        ]
        self.ingest_at(rows, iso(days=1, hours=5), 'b1')
        self.order_at(order(opened_at=iso(days=0, hours=0), closed_at=iso(days=0, hours=6)), iso(days=0, hours=8))
        result = builder.build_run(self.db, self.config(components=['engine1'], outlier_multiplier='1.5'))
        sample_block = result['samples'][0]
        self.assertTrue(sample_block['buildable'])
        device = next(block for block in sample_block['features']['devices'] if block['device_id'] == 'dev1')
        self.assertEqual(device['samples'], 4)
        self.assertEqual(Decimal(device['mean']), Decimal('33'))
        self.assertEqual(device['quantiles']['0.5'], '11.5')
        self.assertTrue(Decimal(device['coverage']['coverage_ratio']) > 0)
        self.assertEqual(Decimal(device['outlier_ratio']), Decimal('0.25'))
        self.assertEqual(device['outlier_method'], 'iqr')
        nearest = device['nearest_maintenance']
        self.assertEqual(nearest['order_id'], 'order1')
        expected_seconds = (
            dt.datetime.fromisoformat(iso(days=3).replace('Z', '+00:00'))
            - dt.datetime.fromisoformat(iso(days=0, hours=6).replace('Z', '+00:00'))
        ).total_seconds()
        self.assertEqual(nearest['distance_seconds'], expected_seconds)
        self.assertIn('cal1@1', device['lineage']['calibration_revisions'])

    def test_feature_ignores_event_committed_after_knowledge_cutoff(self):
        self.ingest_at([sample(1, '10', time=iso(days=1))], iso(days=1, hours=2), 'early')
        self.ingest_at([sample(2, '180', time=iso(days=1, hours=4))], iso(days=6), 'late')
        result = builder.build_run(self.db, self.config(cutoff=iso(days=4), components=['engine1']))
        device = next(block for block in result['samples'][0]['features']['devices'] if block['device_id'] == 'dev1')
        self.assertEqual(device['samples'], 1)
        self.assertEqual(device['mean'], '10')
        # With the later cutoff the row becomes visible and the mean changes.
        result2 = builder.build_run(self.db, self.config(cutoff=iso(days=7), components=['engine1']))
        device2 = next(block for block in result2['samples'][0]['features']['devices'] if block['device_id'] == 'dev1')
        self.assertEqual(device2['samples'], 2)

    def test_post_hoc_calibration_correction_cannot_change_historical_features(self):
        self.ingest_at([sample(1, '10', time=iso(days=1))], iso(days=1, hours=2), 'b1')
        before = builder.build_run(self.db, self.config(cutoff=iso(days=4), components=['engine1']))
        value_before = next(
            block for block in before['samples'][0]['features']['devices'] if block['device_id'] == 'dev1'
        )['mean']
        self.assertEqual(value_before, '10')
        self.clock.point = dt.datetime.fromisoformat(iso(days=5).replace('Z', '+00:00'))
        devices.correct_calibration(self.db, 'cal1', 1, '2', '0', 0)
        after = builder.build_run(self.db, self.config(cutoff=iso(days=4), components=['engine1']))
        value_after = next(
            block for block in after['samples'][0]['features']['devices'] if block['device_id'] == 'dev1'
        )['mean']
        self.assertEqual(value_after, '10')
        lineage = next(
            block for block in after['samples'][0]['features']['devices'] if block['device_id'] == 'dev1'
        )['lineage']
        self.assertEqual(lineage['calibration_parameters'][0]['revision'], 1)


class LabelTests(KnowledgeCutoffCase):
    def base_config(self, point, cutoff_label=iso(days=10), **changes):
        config = {
            'aircraft': 'plane1',
            'prediction_points': [point],
            'lookback_seconds': 86400,
            'horizon_seconds': 2 * 86400,
            'knowledge_cutoff': point,
            'label_cutoff': cutoff_label,
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        config.update(changes)
        return config

    def test_censored_horizon_is_not_negative(self):
        point = iso(days=3)
        # Label cutoff arrives before the two-day horizon fully elapses.
        result = builder.build_run(self.db, self.base_config(point, cutoff_label=iso(days=3, hours=12)))
        sample_block = result['samples'][0]
        self.assertFalse(sample_block['buildable'])
        self.assertIsNone(sample_block['label'])
        self.assertEqual(sample_block['label_state'], 'censored')
        self.assertEqual(sample_block['reasons'][0]['code'], 'right_censored')
        self.assertEqual(result['summary']['negative'], 0)
        self.assertEqual(result['summary']['right_censored'], 1)

    def test_positive_then_negative_after_frozen_horizon(self):
        point = iso(days=3)
        config = self.base_config(point)
        negative = builder.build_run(self.db, config)
        self.assertEqual(negative['samples'][0]['label_state'], 'negative')
        # A work order inside the horizon, committed before the label cutoff.
        self.order_at(order(opened_at=iso(days=3, hours=6), closed_at=iso(days=3, hours=8)), iso(days=4))
        positive = builder.build_run(self.db, config)
        self.assertEqual(positive['samples'][0]['label'], True)
        self.assertEqual(positive['samples'][0]['lineage']['label_trigger']['order_id'], 'order1')

    def test_order_committed_after_label_cutoff_cannot_create_label(self):
        point = iso(days=3)
        self.order_at(order(opened_at=iso(days=3, hours=6), closed_at=iso(days=3, hours=8)), iso(days=11))
        result = builder.build_run(self.db, self.base_config(point, cutoff_label=iso(days=10)))
        self.assertEqual(result['samples'][0]['label_state'], 'negative')

    def test_issue_and_state_conditions_filter_labels(self):
        point = iso(days=3)
        self.order_at(order(opened_at=iso(days=3, hours=6), closed_at=iso(days=3, hours=8),
                            issue='hydraulic leak', state='open'), iso(days=4))
        closed_only = builder.build_run(self.db, self.base_config(point))
        self.assertEqual(closed_only['samples'][0]['label_state'], 'negative')
        issue_scoped = builder.build_run(self.db, self.base_config(
            point, label_condition={'components': ['engine1'], 'states': ['open', 'closed'], 'issues': ['fuel']}))
        self.assertEqual(issue_scoped['samples'][0]['label_state'], 'negative')
        matching = builder.build_run(self.db, self.base_config(
            point, label_condition={'components': ['engine1'], 'states': ['open', 'closed'], 'issues': ['hydraulic']}))
        self.assertEqual(matching['samples'][0]['label'], True)


class BuildFailureTests(KnowledgeCutoffCase):
    def test_missing_historical_calibration_evidence_fails_closed(self):
        # Manually delete the frozen revision to simulate missing evidence:
        # the builder must refuse rather than read the mutable current row.
        self.ingest([sample(1, '10', time=iso(days=1))], batch_id='b1')
        self.db.execute('DELETE FROM calibration_versions WHERE calibration_id=?', ('cal1',))
        config = {
            'aircraft': 'plane1',
            'prediction_points': [iso(days=3)],
            'lookback_seconds': 86400 * 3,
            'horizon_seconds': 86400 * 2,
            'knowledge_cutoff': iso(days=4),
            'label_cutoff': iso(days=10),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        result = builder.build_run(self.db, config)
        sample_block = result['samples'][0]
        self.assertFalse(sample_block['buildable'])
        self.assertEqual(sample_block['label_state'], 'unbuildable')
        self.assertEqual(sample_block['reasons'][0]['code'], 'historical_evidence_missing')
        self.assertIsNone(sample_block['features'])

    def test_cutoff_ordering_is_validated(self):
        config = {
            'aircraft': 'plane1',
            'prediction_points': [iso(days=3)],
            'lookback_seconds': 86400,
            'horizon_seconds': 86400,
            'knowledge_cutoff': iso(days=6),
            'label_cutoff': iso(days=5),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        with self.assertRaises(PipelineError) as error:
            builder.build_run(self.db, config)
        self.assertEqual(error.exception.code, 'cutoff_order')


class PersistenceTests(KnowledgeCutoffCase):
    def config(self):
        return {
            'aircraft': 'plane1',
            'prediction_points': [iso(days=3)],
            'lookback_seconds': 86400 * 2,
            'horizon_seconds': 86400 * 2,
            'knowledge_cutoff': iso(days=4),
            'label_cutoff': iso(days=10),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }

    def test_run_is_stored_with_digest_and_reloadable(self):
        self.ingest([sample(1, '10', time=iso(days=1))], batch_id='b1')
        saved = builder.save_run(self.db, 'run1', self.config())
        self.assertEqual(saved['summary']['candidates'], 1)
        loaded = store.get_run(self.db, 'run1')
        self.assertEqual(loaded['digest'], saved['run']['digest'])
        self.assertEqual(loaded['samples'][0]['sample_id'], saved['run']['samples'][0]['sample_id'])
        with self.assertRaises(PipelineError):
            builder.save_run(self.db, 'run1', self.config())

    def test_tampering_is_detected_by_digest(self):
        self.ingest([sample(1, '10', time=iso(days=1))], batch_id='b1')
        builder.save_run(self.db, 'run1', self.config())
        self.db.execute("UPDATE training_samples SET label=1 WHERE run_id='run1'")
        with self.assertRaises(PipelineError) as error:
            store.get_run(self.db, 'run1')
        self.assertEqual(error.exception.code, 'training_run_corrupt')


class SplitTests(KnowledgeCutoffCase):
    def _register_third_aircraft(self):
        devices.register(self.db, 'dev3', 'plane3', 'engine1', 'temperature', 'C')
        devices.add_alias(self.db, 'feed1', 'sensor3', 'dev3', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        devices.add_calibration(self.db, 'cal3', 'dev3', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z')
        rules.configure(self.db, 'dev3', minimum='-100', maximum='200')

    def _multi_config(self, points_by_aircraft, **changes):
        config = {
            'aircrafts': ['plane1', 'plane2', 'plane3'],
            'prediction_points_by_aircraft': points_by_aircraft,
            'lookback_seconds': 86400,
            'horizon_seconds': 86400,
            'knowledge_cutoff': iso(days=25),
            'label_cutoff': iso(days=40),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        config.update(changes)
        return config

    def test_split_keeps_each_aircraft_in_one_fold_and_orders_folds_in_time(self):
        self._register_third_aircraft()
        self.ingest([
            sample(1, '10', time=iso(days=1), device='sensor1'),
            sample(2, '10', time=iso(days=1), device='sensor2'),
            sample(3, '10', time=iso(days=1), device='sensor3'),
        ], batch_id='b1')
        config = self._multi_config({
            'plane1': [iso(days=3), iso(days=4)],
            'plane2': [iso(days=8), iso(days=9)],
            'plane3': [iso(days=13), iso(days=14)],
        })
        saved = builder.save_multi_aircraft_run(self.db, 'run-multi', config)
        self.assertEqual(saved['summary']['buildable'], 6)
        manifest = splits.save_manifest(self.db, 'run-multi', 'split1', 3)['manifest']
        aircraft_by_fold = [set(fold['aircraft']) for fold in manifest['folds']]
        # No aircraft is shared across folds.
        self.assertEqual(len(set().union(*aircraft_by_fold)), 3)
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse(aircraft_by_fold[left] & aircraft_by_fold[right])
        # Time ordering: every earlier fold ends before the next begins.
        ordered = [fold for fold in manifest['folds'] if fold['sample_ids']]
        for earlier, later in zip(ordered, ordered[1:]):
            self.assertLessEqual(earlier['max_prediction_point'], later['min_prediction_point'])
        self.assertTrue(splits.verify_manifest(self.db, 'run-multi', 'split1')['ok'])

    def test_overlapping_label_windows_stay_together(self):
        self._register_third_aircraft()
        self.ingest([
            sample(1, '10', time=iso(days=1), device='sensor1'),
            sample(2, '10', time=iso(days=1), device='sensor2'),
            sample(3, '10', time=iso(days=1), device='sensor3'),
        ], batch_id='b1')
        # plane1 has two overlapping horizons; plane2 and plane3 give the two
        # remaining, time-later fold groups.
        config = self._multi_config({
            'plane1': [iso(days=3), iso(days=3, hours=12)],
            'plane2': [iso(days=8)],
            'plane3': [iso(days=13)],
        }, horizon_seconds=2 * 86400, knowledge_cutoff=iso(days=20), label_cutoff=iso(days=30))
        saved = builder.save_multi_aircraft_run(self.db, 'run-overlap', config)
        result_samples = saved['run']['samples']
        component_sizes = sorted(len(component) for component in splits._overlap_components(result_samples))
        self.assertIn(2, component_sizes)
        manifest = splits.save_manifest(self.db, 'run-overlap', 'split1', 3)['manifest']
        self.assertEqual(manifest['sample_count'], 4)
        fold_of = {}
        for fold in manifest['folds']:
            for sample_id in fold['sample_ids']:
                fold_of[sample_id] = fold['fold']
        overlapping_ids = {
            sample['sample_id'] for sample in result_samples
            if sample['aircraft'] == 'plane1'
        }
        # Both overlapping plane1 samples stay in a single fold.
        self.assertEqual(len({fold_of[sample_id] for sample_id in overlapping_ids}), 1)

    def test_censored_samples_are_excluded_and_listed_in_manifest(self):
        config = {
            'aircraft': 'plane1',
            'prediction_points': [iso(days=3)],
            'lookback_seconds': 86400,
            'horizon_seconds': 2 * 86400,
            'knowledge_cutoff': iso(days=3),
            'label_cutoff': iso(days=3, hours=6),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        builder.save_run(self.db, 'run-censored', config)
        with self.assertRaises(PipelineError) as error:
            splits.make_manifest(self.db, 'run-censored', 2)
        self.assertEqual(error.exception.code, 'no_buildable_samples')
        # The exclusion list is still available through the component builder.
        run = store.get_run(self.db, 'run-censored')
        _, excluded = splits._buildable_samples(run)
        self.assertEqual(excluded[0]['reason'], 'censored')

    def test_manifest_tampering_is_detected(self):
        self._register_third_aircraft()
        self.ingest([
            sample(1, '10', time=iso(days=1), device='sensor1'),
            sample(2, '10', time=iso(days=1), device='sensor2'),
            sample(3, '10', time=iso(days=1), device='sensor3'),
        ], batch_id='b1')
        config = self._multi_config({
            'plane1': [iso(days=3)], 'plane2': [iso(days=8)], 'plane3': [iso(days=13)],
        })
        builder.save_multi_aircraft_run(self.db, 'run-multi', config)
        splits.save_manifest(self.db, 'run-multi', 'split1', 3)
        self.db.execute("UPDATE training_splits SET manifest=manifest WHERE split_id='split1'")
        self.assertTrue(splits.verify_manifest(self.db, 'run-multi', 'split1')['ok'])
        row = store.get_split(self.db, 'run-multi', 'split1')
        row['manifest']['folds'][0]['sample_ids'].append('forged')
        stored = self.db.one("SELECT digest FROM training_splits WHERE split_id='split1'")
        self.assertNotEqual(digest(row['manifest']), stored['digest'])

import datetime as dt

from hangar_ingest.api import Application
from hangar_ingest.types import PipelineError
from tests.support import sample
from tests.training.test_point_in_time import KnowledgeCutoffCase, iso


class TrainingApplicationTests(KnowledgeCutoffCase):
    def setUp(self):
        super().setUp()
        self.app = Application(self.db)

    def _config(self, **changes):
        config = {
            'aircraft': 'plane1',
            'prediction_points': [iso(days=3)],
            'lookback_seconds': 2 * 86400,
            'horizon_seconds': 2 * 86400,
            'knowledge_cutoff': iso(days=4),
            'label_cutoff': iso(days=10),
            'label_condition': {'components': ['engine1'], 'states': ['closed']},
        }
        config.update(changes)
        return config

    def test_training_operations_are_registered_and_persist(self):
        self.ingest([sample(1, '10', time=iso(days=1))], batch_id='b1')
        preview = self.app.call('training.preview', {'config': self._config()})
        self.assertEqual(preview['summary']['candidates'], 1)
        self.app.call('training.save', {'run_id': 'run1', 'config': self._config()})
        loaded = self.app.call('training.get', {'run_id': 'run1'})
        self.assertEqual(len(loaded['samples']), 1)
        buildable = self.app.call('training.samples', {'run_id': 'run1', 'buildable': True})
        self.assertEqual(len(buildable), 1)
        listing = self.app.call('training.list', {})
        self.assertEqual([row['run_id'] for row in listing], ['run1'])

    def test_censored_sample_excluded_from_split_but_listed(self):
        # Knowledge cutoff is the prediction point itself; the label cutoff
        # lands later but before the two-day horizon fully elapses.
        config = self._config(knowledge_cutoff=iso(days=3), label_cutoff=iso(days=3, hours=12))
        self.app.call('training.save', {'run_id': 'run1', 'config': config})
        with self.assertRaises(PipelineError) as error:
            self.app.call('training.split', {'run_id': 'run1', 'split_id': 's1', 'fold_count': 2})
        self.assertEqual(error.exception.code, 'no_buildable_samples')

    def test_unknown_operation_and_bad_params_are_rejected(self):
        with self.assertRaises(PipelineError):
            self.app.call('training.nope', {})
        with self.assertRaises(PipelineError):
            self.app.call('training.preview', {'config': {'aircraft': 'plane1'}})

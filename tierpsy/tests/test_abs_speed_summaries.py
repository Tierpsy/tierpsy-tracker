import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tierpsy.features.tierpsy_features.summary_stats import (
    _get_cross_subdivided_features,
    get_df_quantiles,
    get_summary_stats,
)


SPEED_FEATURES = ['speed_midbody', 'speed_head', 'speed_tail']


class TestAbsoluteSpeedSummaries(unittest.TestCase):

    def setUp(self):
        self.timeseries = pd.DataFrame({
            'speed_midbody': [-100.0, 20.0, 30.0, -40.0],
            'speed_head': [-90.0, 10.0, 50.0, -30.0],
            'speed_tail': [-110.0, 30.0, 20.0, -50.0],
            'behavioural_states': [2, 2, 2, 2],
            'motion_mode': [-1, 1, 1, -1],
        })

    def test_state_median_uses_absolute_values_before_aggregation(self):
        original = self.timeseries.copy()

        result = get_df_quantiles(
            self.timeseries,
            feats2check=SPEED_FEATURES,
            feats2abs=SPEED_FEATURES,
            feats2norm=[],
            subdivision_dict={'behavioural_states': SPEED_FEATURES},
            is_abs_ventral=True,
        )

        self.assertEqual(result['speed_midbody_w_roaming_abs_50th'], 35.0)
        pd.testing.assert_frame_equal(self.timeseries, original)

    def test_directional_summaries_use_state_and_motion_intersection(self):
        subdivided = _get_cross_subdivided_features(
            self.timeseries,
            timeseries_cols=SPEED_FEATURES,
            subdivision_values={
                'behavioural_states': (1, 2, 3),
                'motion_mode': (-1, 1),
            },
        )
        result = get_df_quantiles(
            subdivided,
            feats2check=subdivided.columns.tolist(),
            feats2abs=SPEED_FEATURES,
            feats2norm=[],
            subdivision_dict={},
            is_remove_subdivided=False,
            is_abs_ventral=True,
        )

        self.assertEqual(
            result['speed_midbody_w_roaming_w_backward_abs_50th'], 70.0)
        self.assertEqual(
            result['speed_midbody_w_roaming_w_forward_abs_50th'], 25.0)
        self.assertTrue(np.isnan(
            result['speed_midbody_w_sprinting_w_forward_abs_50th']))

    def test_summary_stats_exposes_absolute_and_directional_speeds(self):
        timeseries = self.timeseries.copy()
        timeseries['worm_index'] = 1
        timeseries['timestamp'] = np.arange(len(timeseries))
        timeseries['length'] = 1000.0
        timeseries['behavioural_states'] = \
            timeseries['behavioural_states'].astype(float)
        timeseries['motion_mode'] = timeseries['motion_mode'].astype(float)
        timeseries['turn'] = 0.0
        selected = [
            'speed_midbody_w_roaming_abs_50th',
            'speed_midbody_w_roaming_w_forward_abs_50th',
            'speed_midbody_w_roaming_w_backward_abs_50th',
        ]

        result = get_summary_stats(
            timeseries, fps=25, selected_feat=selected)

        self.assertEqual(result[selected[0]], 35.0)
        self.assertEqual(result[selected[1]], 25.0)
        self.assertEqual(result[selected[2]], 70.0)

    def test_tierpsy_levels_select_only_absolute_state_speeds(self):
        feature_set_dir = (
            Path(__file__).parents[1] / 'extras' / 'feat_sets')
        all_feature_names = set(
            (feature_set_dir / 'tierpsy_features_all_names.csv')
            .read_text(encoding='utf-8-sig').splitlines())
        for level in range(1, 5):
            names = (feature_set_dir / 'tierpsy_level_{}.csv'.format(level)) \
                .read_text().splitlines()
            speeds = [
                name for name in names
                if name.startswith('speed_midbody_')
                or name.startswith('speed_head_')
                or name.startswith('speed_tail_')
            ]

            self.assertTrue(speeds)
            self.assertTrue(all('_abs_' in name for name in speeds))
            self.assertTrue(any('_w_forward_abs_' in name for name in speeds))
            self.assertTrue(any('_w_backward_abs_' in name for name in speeds))
            self.assertTrue(set(speeds).issubset(all_feature_names))


if __name__ == '__main__':
    unittest.main()

# test_functions.py
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from astropy.timeseries import LombScargle

# Import functions from your script
import sys, os
# Define home directory
cmd_folder = os.getenv("RESEARCH") + "Roman_lc_gen"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from functions import (
    phase_refine,
    add_last_point,
    T_0_fixer,
    stats_binning,
    find_valid_rows,
    convert_phase_to_days,
    l2_distance,
    get_metrics,
    prep_input,
    create_multi_phase_repeating_folded_lc # Needed by prep_input
    # Note: Testing functions like read_fits, prep_gp, fit_gp, predict_gp, run_all
    # would require more complex mocking of file I/O and external libraries (george, astropy.io.fits)
    # and potentially dummy data files.
)

# Mock the gp_model_params.info dictionary if it's used directly
# If it's passed as an argument 'info_', we handle it in the test calls.
try:
    import gp_model_params
    # If gp_model_params exists but doesn't have info, create a dummy one
    if not hasattr(gp_model_params, 'info'):
        gp_model_params.info = {'RRLYR': {'p_range': [0.2, 1.0], 'n_phs': 5}} # Example
except ImportError:
    # If gp_model_params doesn't exist at all, create a mock module
    mock_gp_params = MagicMock()
    mock_gp_params.info = {'RRLYR': {'p_range': [0.2, 1.0], 'n_phs': 5}} # Example
    import sys
    sys.modules['gp_model_params'] = mock_gp_params


class TestFunctions(unittest.TestCase):

    def test_phase_refine(self):
        """Test period refinement using LombScargle (mocked)."""
        # Mock LombScargle
        mock_ls_instance = MagicMock()
        # Define the power spectrum the mock should return
        # Let's say the peak power is at the 3rd frequency -> 3rd period
        mock_power = np.array([0.1, 0.5, 1.0, 0.4, 0.2])
        mock_ls_instance.power.return_value = mock_power

        mock_lombscargle = MagicMock(return_value=mock_ls_instance)

        # Patch LombScargle within the functions module
        with patch('functions.LombScargle', mock_lombscargle):
            t = np.linspace(0, 10, 50)
            m = np.sin(2 * np.pi * t / 0.5) # Signal with period 0.5
            e = np.ones_like(t) * 0.1
            p_range = [0.3, 0.7] # Range includes true period

            # Expected period based on mock_power peak (index 2)
            expected_periods = np.linspace(p_range[0] - (p_range[0] / 10), p_range[1] + (p_range[1] / 10), 100)
            expected_p = expected_periods[2] # Corresponds to index 2 where power is max

            phase, p = phase_refine(t, m, e, p_range)

            # Assertions
            mock_lombscargle.assert_called_once_with(t, m, e, nterms=5)
            mock_ls_instance.power.assert_called_once()
            self.assertEqual(p, expected_p)
            self.assertEqual(len(phase), len(t))
            self.assertTrue(np.all(phase >= 0) and np.all(phase < 1))
            np.testing.assert_array_almost_equal(phase, (t/expected_p)%1)

    def test_add_last_point(self):
        """Test adding a point based on the phase closest to a fixed time."""
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4])
        m = np.array([10, 11, 12, 13, 14, 10.5, 11.5, 12.5, 13.5, 14.5])
        e = np.ones_like(t) * 0.1
        phase = (t / 0.5) % 1 # Example period 0.5
        band = ['I'] * len(t)
        df = pd.DataFrame({'t': t, 'm': m, 'e': e, 'phase': phase, 'band': band})
        p = 0.5

        # Hardcoded time: 1713.99. Phase = (1713.99 / 0.5) % 1 = 3427.98 % 1 = 0.98
        # Closest phase in df: index 4 (t=0.4, phase=0.8) or index 9 (t=1.4, phase=0.8)
        # np.argmin will pick the first one, index 4
        expected_added_row = df.iloc[4]

        df_new = add_last_point(df.copy(), p) # Use copy to avoid modifying original

        # Assertions
        self.assertEqual(len(df_new), len(df) + 1)
        self.assertEqual(df_new.iloc[-1]['t'], 1713.99)
        self.assertEqual(df_new.iloc[-1]['m'], expected_added_row['m'])
        self.assertEqual(df_new.iloc[-1]['e'], expected_added_row['e'])
        # self.assertEqual(df_new.iloc[-1]['phase'], expected_added_row['phase']) # Phase is recalculated? Check function logic
        self.assertEqual(df_new.iloc[-1]['phase'], df.phase[4]) # Phase seems copied directly
        self.assertEqual(df_new.iloc[-1]['band'], expected_added_row['band'])
        self.assertTrue(pd.api.types.is_float_dtype(df_new['t']))
        self.assertTrue(pd.api.types.is_float_dtype(df_new['m']))
        self.assertTrue(pd.api.types.is_float_dtype(df_new['e']))
        self.assertTrue(pd.api.types.is_float_dtype(df_new['phase']))

    def test_t_0_fixer(self):
        """Test shifting phase so minimum magnitude is near phase 0.25."""
        p = 1.0
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) # Times
        m = np.array([10, 11, 12, 9, 10.1]) # Min magnitude at t=0.75
        phase_initial = (t / p) % 1 # [0.0, 0.25, 0.5, 0.75, 0.0]

        # T0 = t[argmin(m)] = 0.75
        # phase0 = (T0/p)%1 = 0.75
        # temp = phase0 - 0.25 = 0.75 - 0.25 = 0.5
        # Expected phase = (phase_initial - temp) % 1
        # [0.0 - 0.5 -> -0.5 -> 0.5]
        # [0.25 - 0.5 -> -0.25 -> 0.75]
        # [0.5 - 0.5 -> 0.0]
        # [0.75 - 0.5 -> 0.25] # Minimum is now at 0.25
        # [0.0 - 0.5 -> -0.5 -> 0.5]
        expected_phase = np.array([0.5, 0.75, 0.0, 0.25, 0.5])

        phase_fixed = T_0_fixer(t, m, p, phase_initial.copy()) # Use copy

        np.testing.assert_array_almost_equal(phase_fixed, expected_phase)
        self.assertTrue(np.all(phase_fixed >= 0) and np.all(phase_fixed <= 1))

    def test_stats_binning(self):
        """Test statistical binning of data."""
        x = np.linspace(0, 9.9, 100)
        y = np.sin(x) + 10 # Example data
        e = np.random.rand(100) * 0.1 # Example errors
        bins = 10

        bin_middles, bin_means, bin_errors = stats_binning(x, y, e, bins=bins)

        # Assertions
        self.assertEqual(len(bin_middles), bins)
        self.assertEqual(len(bin_means), bins)
        self.assertEqual(len(bin_errors), bins)

        # Check middle calculation (approximate)
        bin_width = (np.max(x) - np.min(x)) / bins
        expected_first_middle = np.min(x) + bin_width / 2
        self.assertAlmostEqual(bin_middles[0], expected_first_middle, places=5)

        # Check one bin's mean manually (e.g., first bin: x=0 to 0.99)
        mask = (x >= 0) & (x < bin_width)
        expected_first_mean = np.mean(y[mask])
        self.assertAlmostEqual(bin_means[0], expected_first_mean, places=5)

        # Check one bin's error manually (e.g., first bin)
        expected_first_error = np.median(e[mask]) / 50
        # Handle cases where a bin might be empty, though unlikely here
        if not np.isnan(expected_first_error):
             self.assertAlmostEqual(bin_errors[0], expected_first_error, places=5)
        else:
             self.assertTrue(np.isnan(bin_errors[0])) # If bin was empty


    def test_find_valid_rows_conditions(self):
        """Test the logic for finding valid rows based on metric thresholds."""
        # [std_all, l2_bin_reg, l2_bin, l2_part1, l2_part2, l2_part3]
        matrix = np.array([
            [1.0, 0.005, 0.001, 0.0005, 0.0006, 0.0007], # Valid (level 4)
            [3.0, 0.005, 0.001, 0.0005, 0.0006, 0.0007], # Invalid (std_all > 2)
            [1.0, 0.050, 0.001, 0.0005, 0.0006, 0.0007], # Invalid (l2_bin_reg > threshold_std)
            [1.0, 0.005, 0.050, 0.0005, 0.0006, 0.0007], # Invalid (l2_bin > threshold)
            [1.0, 0.005, 0.001, 0.0500, 0.0006, 0.0007], # Invalid (l2_part1 > threshold)
            [1.0, 0.005, 0.001, 0.0005, 0.0500, 0.0007], # Invalid (l2_part2 > threshold)
            [1.0, 0.005, 0.001, 0.0005, 0.0006, 0.0500], # Invalid (l2_part3 > threshold)
            [1.0, 0.005, 0.001, 0.0001, 0.0001, 0.0090], # Valid (level 4) - testing std ratio
            [1.0, 0.005, 0.001, 0.0001, 0.001, 0.008],   # Valid (level 4)
            [1.0, 0.005, 0.001, 0.00001, 0.001, 0.008], # Valid (level 4) - close std ratio
        ])
        threshold = 0.01
        threshold_std = 0.01

        valid_indices_l4 = find_valid_rows(matrix, threshold, threshold_std, level=4)
        valid_indices_l3 = find_valid_rows(matrix, threshold, threshold_std, level=3) # Ignores std_ratio check

        np.testing.assert_array_equal(valid_indices_l4, np.array([0, 7, 8, 9]))
        # Level 3 would potentially include rows where std_ratio failed, if others passed
        # In this specific matrix, level 3 results are the same as level 4
        np.testing.assert_array_equal(valid_indices_l3, np.array([0, 7, 8, 9]))


    def test_find_valid_rows_edge_cases(self):
        """Test find_valid_rows with empty or NaN input."""
        empty_matrix = np.empty((0, 6))
        nan_matrix = np.array([[1.0, 0.005, np.nan, 0.001, 0.001, 0.001]])

        self.assertEqual(len(find_valid_rows(empty_matrix)), 0)
        self.assertEqual(len(find_valid_rows(nan_matrix)), 0) # NaN should fail the < threshold check

    def test_convert_phase_to_days(self):
        """Test converting phase back to time units."""
        # Note: The function currently ignores t_max and the replication logic (num_x_n) seems commented out/simplified
        # Testing the current implementation: converted_x = phs_array * period * n_phases
        phs_array = np.array([0.0, 0.25, 0.5, 0.75])
        y_array = np.ones_like(phs_array) * 10
        e_array = np.ones_like(phs_array) * 0.1
        n_phases = 5.0
        period = 2.0
        t_max = 50.0 # Currently unused in the effective code path

        expected_x = phs_array * period * n_phases # 0.0, 2.5, 5.0, 7.5

        conv_x, conv_y, conv_e = convert_phase_to_days(phs_array, y_array, e_array, n_phases, period, t_max)

        np.testing.assert_array_almost_equal(conv_x, expected_x)
        np.testing.assert_array_equal(conv_y, y_array) # y should be unchanged
        np.testing.assert_array_equal(conv_e, e_array) # e should be unchanged

    def test_l2_distance(self):
        """Test L2 distance calculation between two potentially non-aligned datasets."""
        x1 = np.linspace(0, 10, 11)
        y1 = np.sin(x1)
        x2 = np.linspace(0.5, 10.5, 11) # Shifted x
        y2 = np.sin(x2 - 0.5) # Same underlying function, shifted

        # Overlap is [0.5, 10.0]
        # Since y2(t) = sin(t), and y1(t) = sin(t), they should be identical in overlap
        l2_dist = l2_distance(x1, y1, x2, y2)
        self.assertAlmostEqual(l2_dist, 0.0, places=5)

        # Test with different functions
        y3 = np.cos(x1)
        l2_dist_diff = l2_distance(x1, y1, x1, y3) # Same x, different y
        self.assertGreater(l2_dist_diff, 0.1) # Should be non-zero

        # Test non-overlapping
        x4 = np.linspace(11, 20, 10)
        y4 = np.sin(x4)
        l2_dist_no_overlap = l2_distance(x1, y1, x4, y4)
        # Depending on implementation details (linspace with num=1?), might be 0 or NaN/error.
        # Current linspace(min, max, 500) handles this reasonably. If min > max, it might produce a single point or error.
        # Let's expect NaN or 0. Testing for NaN is safer.
        # self.assertTrue(np.isnan(l2_dist_no_overlap) or l2_dist_no_overlap == 0.0)
        # Update: Based on the code, if x_min > x_max, common_x becomes [], sum([])=0, sqrt(0)=0.
        self.assertAlmostEqual(l2_dist_no_overlap, 0.0, places=6)


    @patch('functions.l2_distance') # Mock l2_distance for get_metrics test
    def test_get_metrics(self, mock_l2_dist):
        """Test the calculation of various fit metrics."""
        mock_l2_dist.return_value = 0.005 # Provide a mock return value for l2_distance

        # Simple data where metrics are easy to calculate
        x_binned = np.linspace(0, 9, 10)
        y_binned = np.ones(10) * 5.0
        gp_y_binned_fit = np.ones(10) * 5.0 # Perfect fit for y_binned
        regular_sampling_fit = np.linspace(0, 9, 50)
        gp_y_regular_fit = np.ones(50) * 5.0 # Perfect fit for regular sampling too

        metrics = get_metrics(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit, y_binned)

        # Expected: [std_ratio, l2_dist, all_lc_metric, part1, part2, part3]
        # std(y_binned) = 0, std(gp_y_binned_fit) = 0 -> division by zero -> NaN or Inf expected for std_ratio
        # all_lc_metric = sum((5-5)**2)/10 = 0
        # part1/2/3 metrics = mean((5-5)**2) = 0
        # l2_dist is mocked to 0.005
        self.assertTrue(np.isnan(metrics[0]) or np.isinf(metrics[0])) # std_ratio check
        self.assertAlmostEqual(metrics[1], 0.005) # l2_bin_reg (mocked)
        self.assertAlmostEqual(metrics[2], 0.0)   # l2_bin
        self.assertAlmostEqual(metrics[3], 0.0)   # l2_bin_part1
        self.assertAlmostEqual(metrics[4], 0.0)   # l2_bin_part2
        self.assertAlmostEqual(metrics[5], 0.0)   # l2_bin_part3
        mock_l2_dist.assert_called_once_with(x_binned, gp_y_binned_fit, regular_sampling_fit, gp_y_regular_fit)

        # Test with imperfect fit
        gp_y_binned_fit_imperfect = y_binned + 0.1
        metrics_imp = get_metrics(x_binned, gp_y_binned_fit_imperfect, regular_sampling_fit, gp_y_regular_fit, y_binned)
        expected_all_lc_metric = np.sum((y_binned - gp_y_binned_fit_imperfect)**2) / len(y_binned) # (0.1**2)*10 / 10 = 0.01
        expected_part_metric = np.mean((y_binned[0:3] - gp_y_binned_fit_imperfect[0:3])**2) # mean(0.1**2) = 0.01
        self.assertAlmostEqual(metrics_imp[0], 1.0) # std(y)/std(y+c) = std(y)/(std(y)) = 1.0 (if std(y) != 0), need non-constant y
        self.assertAlmostEqual(metrics_imp[2], expected_all_lc_metric)
        self.assertAlmostEqual(metrics_imp[3], expected_part_metric)
        # Need non-constant y for std ratio
        y_binned_nc = np.arange(10)
        gp_y_binned_fit_nc = y_binned_nc + 1
        metrics_nc = get_metrics(x_binned, gp_y_binned_fit_nc, regular_sampling_fit, gp_y_regular_fit, y_binned_nc)
        self.assertAlmostEqual(metrics_nc[0], 1.0) # std(y)/std(y+c) still 1.0

    # Patch the dependency 'create_multi_phase_repeating_folded_lc'
    @patch('functions.create_multi_phase_repeating_folded_lc')
    def test_prep_input(self, mock_create_multi_phase):
        """Test input preparation logic for finite and infinite n_phases."""
        # Shared dummy data
        t = np.array([5, 1, 4, 2, 3])
        m = np.array([15, 11, 14, 12, 13])
        e = np.ones_like(t) * 0.1
        df = pd.DataFrame({'t': t, 'm': m, 'e': e})
        p = 1.0 # Dummy period
        t_max = 10.0 # Dummy t_max

        # Case 1: Finite n_phases
        info_finite = {'n_phs': 5}
        # Mock the return value of the patched function
        mock_x = np.array([0.8, 0.2, 0.6, 0.4, 0.0]) # Unsorted phase-like data
        mock_y = np.array([15, 11, 14, 12, 13])
        mock_e = np.ones_like(mock_y) * 0.2
        mock_create_multi_phase.return_value = (mock_x, mock_y, mock_e)

        x_fin, y_fin, e_fin = prep_input(df, 'RRLYR', p, info_finite, t_max)

        mock_create_multi_phase.assert_called_once_with(info_finite['n_phs'], p, df, t_max)
        # Check if output is sorted by x
        np.testing.assert_array_equal(x_fin, np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
        np.testing.assert_array_equal(y_fin, np.array([13, 11, 12, 14, 15])) # y sorted according to x
        np.testing.assert_array_equal(e_fin, np.array([0.2, 0.2, 0.2, 0.2, 0.2])) # e sorted according to x

        # Case 2: Infinite n_phases
        info_infinite = {'n_phs': np.inf}
        mock_create_multi_phase.reset_mock() # Reset call count for next test

        x_inf, y_inf, e_inf = prep_input(df, 'RRLYR', p, info_infinite, t_max)

        mock_create_multi_phase.assert_not_called() # Should not be called for infinite n_phs
        # Check if output is sorted by original time t
        np.testing.assert_array_equal(x_inf, np.array([1, 2, 3, 4, 5])) # Sorted t
        np.testing.assert_array_equal(y_inf, np.array([11, 12, 13, 14, 15])) # m sorted according to t
        np.testing.assert_array_equal(e_inf, np.array([0.1, 0.1, 0.1, 0.1, 0.1])) # e sorted according to t


# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Need argv for some environments like notebooks
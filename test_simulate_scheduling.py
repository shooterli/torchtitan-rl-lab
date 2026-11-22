"""Checks for accounting, event ordering, and scheduler resource limits."""

from dataclasses import replace
import unittest

from simulate_scheduling import Settings, simulate


class SchedulingTests(unittest.TestCase):
    def test_serial_baseline_matches_closed_form(self):
        settings = Settings()
        result = simulate(settings, 0)
        expected = (settings.groups * (settings.generation_time + settings.train_time)
                    + settings.generation_time * (settings.slow_multiplier - 1))
        self.assertEqual(result["simulated_seconds"], expected)
        self.assertEqual(result["max_policy_age"], 0)

    def test_three_group_straggler_timeline(self):
        settings = Settings(groups=3, slow_multiplier=3)
        greedy = simulate(settings, 1)
        fifo = simulate(replace(settings, window_groups=1), 1)
        self.assertEqual([row["group_id"] for row in greedy["trace"]], [1, 2, 0])
        self.assertEqual(greedy["simulated_seconds"], 7)
        self.assertEqual(greedy["max_policy_age"], 2)
        self.assertEqual([row["group_id"] for row in fifo["trace"]], [0, 1, 2])
        self.assertEqual(fifo["simulated_seconds"], 10)
        self.assertEqual(fifo["max_policy_age"], 1)

    def test_accounting_and_resource_limits(self):
        for groups in (1, 3, 24):
            for workers in (1, 2, 4):
                for target in (0, 1, 3):
                    for window in (0, 1, 2):
                        with self.subTest(groups=groups, workers=workers,
                                          target=target, window=window):
                            settings = Settings(groups=groups, workers=workers,
                                                window_groups=window)
                            result = simulate(settings, target)
                            trace = result["trace"]
                            self.assertEqual(sorted(row["group_id"] for row in trace),
                                             list(range(groups)))
                            self.assertLessEqual(result["peak_active"], target + 1)
                            self.assertEqual(result["simulated_seconds"],
                                             result["batch_wait_seconds"] + groups)
                            events = []
                            for version, row in enumerate(trace):
                                self.assertGreaterEqual(row["train_start"], row["generation_end"])
                                self.assertGreaterEqual(row["policy_age"], 0)
                                self.assertEqual(row["policy_age"], version - row["policy_version"])
                                completed_at_start = sum(
                                    other["train_end"] <= row["generation_start"]
                                    for other in trace
                                )
                                self.assertEqual(row["policy_version"], completed_at_start)
                                self.assertEqual(row["train_end"] - row["train_start"], 1)
                                events.extend([(row["generation_start"], 1),
                                               (row["generation_end"], -1)])
                            running = 0
                            for _, delta in sorted(events):
                                running += delta
                                self.assertLessEqual(running, workers)
                                self.assertGreaterEqual(running, 0)
                            self.assertEqual(running, 0)
                            for previous, current in zip(trace, trace[1:]):
                                self.assertGreaterEqual(current["train_start"], previous["train_end"])


if __name__ == "__main__":
    unittest.main()

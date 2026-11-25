"""Checks for accounting, event ordering, and scheduler resource limits."""

from dataclasses import replace
import unittest
from itertools import product

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

    def test_blocking_sync_and_published_versions(self):
        settings = Settings(groups=3, slow_group=-1, sync_time=2)
        serial = simulate(settings, 0)
        self.assertEqual(serial["simulated_seconds"], 3 * (2 + 1 + 2))
        self.assertEqual(serial["max_policy_age"], 0)
        first, second, third = serial["trace"]
        self.assertEqual(first["train_end"], 3)
        self.assertEqual(first["sync_end"], 5)
        self.assertEqual(second["generation_start"], 5)
        self.assertEqual(second["policy_version"], 1)
        self.assertEqual(third["sync_end"], 15)

        delayed = simulate(replace(settings, workers=1, sync_time=3), 3)
        last_group = next(row for row in delayed["trace"] if row["group_id"] == 2)
        self.assertEqual(last_group["generation_start"], 4)
        self.assertEqual(last_group["policy_version"], 0)

    def test_accounting_and_resource_limits(self):
        for groups, workers, target, window, sync_time in product(
            (1, 3, 24), (1, 2, 4), (0, 1, 3), (0, 1, 2), (0, 1, 3)
        ):
            with self.subTest(groups=groups, workers=workers,
                              target=target, window=window, sync_time=sync_time):
                settings = Settings(groups=groups, workers=workers,
                                    window_groups=window, sync_time=sync_time)
                result = simulate(settings, target)
                trace = result["trace"]
                self.assertEqual(sorted(row["group_id"] for row in trace),
                                 list(range(groups)))
                self.assertLessEqual(result["peak_active"], target + 1)
                self.assertEqual(result["simulated_seconds"],
                                 result["batch_wait_seconds"] + groups + sync_time)
                events = []
                for version, row in enumerate(trace):
                    self.assertGreaterEqual(row["train_start"], row["generation_end"])
                    self.assertGreaterEqual(row["policy_age"], 0)
                    self.assertEqual(row["policy_age"], version - row["policy_version"])
                    completed_at_start = sum(
                        other["sync_end"] <= row["generation_start"]
                        for other in trace
                    )
                    self.assertEqual(row["policy_version"], completed_at_start)
                    self.assertEqual(row["train_end"] - row["train_start"], 1)
                    self.assertEqual(row["sync_end"] - row["train_end"], sync_time)
                    events.extend([(row["generation_start"], 1),
                                   (row["generation_end"], -1)])
                running = 0
                for _, delta in sorted(events):
                    running += delta
                    self.assertLessEqual(running, workers)
                    self.assertGreaterEqual(running, 0)
                self.assertEqual(running, 0)
                for previous, current in zip(trace, trace[1:]):
                    self.assertGreaterEqual(current["train_start"], previous["sync_end"])


if __name__ == "__main__":
    unittest.main()

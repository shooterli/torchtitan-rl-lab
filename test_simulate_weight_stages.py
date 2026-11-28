"""Hand timelines and closed forms for the idealized publication model."""

from dataclasses import replace
from itertools import product
import unittest

from simulate_weight_stages import MODES, Settings, simulate


class WeightStageTests(unittest.TestCase):
    def test_three_step_timeline(self):
        settings = Settings()
        combined = simulate(settings, "combined")
        split = simulate(settings, "split")
        inline = simulate(settings, "inline_push")
        self.assertEqual([row["publish_start"] for row in split["trace"]], [2, 5, 8])
        self.assertEqual([row["optimizer_end"] for row in split["trace"]], [2, 4, 7])
        self.assertEqual([row["publish_start"] for row in combined["trace"]], [2, 6, 10])
        self.assertEqual([row["compute_start"] for row in inline["trace"]], [0, 3, 6])
        self.assertEqual([result["simulated_seconds"] for result in
                          (combined, split, inline)], [13, 11, 11])
        self.assertEqual(split["optimizer_wait_seconds"], 0)
        self.assertEqual(split["publish_wait_seconds"], 2)

    def test_instant_optimizer_and_push_heavy_control(self):
        for compute, push, pull in product((0, 1, 5), (0, 1, 3), (0, 2, 4)):
            settings = Settings(steps=24, compute_time=compute, optimizer_time=0,
                                push_time=push, pull_time=pull)
            self.assertEqual(simulate(settings, "combined")["simulated_seconds"],
                             simulate(settings, "split")["simulated_seconds"])
        settings = replace(Settings(), push_time=3, pull_time=0)
        self.assertEqual(simulate(settings, "combined")["simulated_seconds"], 13)
        self.assertEqual(simulate(settings, "split")["simulated_seconds"], 13)

    def test_closed_forms_dependencies_and_accounting(self):
        for steps, compute, optimizer, push, pull, mode in product(
            (1, 3, 24), (0, 1, 5), (0, 1, 3), (0, 1, 3), (0, 2, 4), MODES
        ):
            with self.subTest(steps=steps, compute=compute, optimizer=optimizer,
                              push=push, pull=pull, mode=mode):
                settings = Settings(steps, compute, optimizer, push, pull)
                result = simulate(settings, mode)
                period = {"combined": max(compute, push + pull) + optimizer,
                          "split": max(compute + optimizer, push + optimizer, push + pull),
                          "inline_push": max(compute + optimizer, pull) + push}[mode]
                self.assertEqual(result["simulated_seconds"],
                                 compute + optimizer + push + pull + (steps - 1) * period)
                self.assertEqual(result["simulated_seconds"],
                                 steps * (compute + optimizer)
                                 + result["optimizer_wait_seconds"]
                                 + result["publish_wait_seconds"]
                                 + result["inline_push_seconds"] + result["final_drain_seconds"])
                previous = None
                for row in result["trace"]:
                    self.assertEqual(row["compute_end"] - row["compute_start"], compute)
                    self.assertEqual(row["optimizer_end"] - row["optimizer_start"], optimizer)
                    self.assertEqual(row["push_end"] - row["publish_start"], push)
                    self.assertEqual(row["pull_end"] - row["push_end"], pull)
                    self.assertGreaterEqual(row["optimizer_wait"], 0)
                    self.assertGreaterEqual(row["publish_wait"], 0)
                    if previous:
                        self.assertGreaterEqual(row["optimizer_start"], previous["push_end"])
                        self.assertGreaterEqual(row["publish_start"], previous["pull_end"])
                        start_key = "push_end" if mode == "inline_push" else "publish_start"
                        self.assertEqual(row["compute_start"], previous[start_key])
                        if mode == "combined":
                            self.assertGreaterEqual(row["optimizer_start"], previous["pull_end"])
                    previous = row
                if mode == "inline_push":
                    self.assertEqual(result["optimizer_wait_seconds"], 0)

    def test_invalid_settings(self):
        for settings in [Settings(steps=0), Settings(compute_time=-1),
                         Settings(optimizer_time=-1), Settings(push_time=-1),
                         Settings(pull_time=-1)]:
            with self.assertRaises(ValueError):
                simulate(settings, "split")
        with self.assertRaises(ValueError):
            simulate(Settings(), "unknown")


if __name__ == "__main__":
    unittest.main()

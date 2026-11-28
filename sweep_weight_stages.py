"""CSV for six fixed-workload push/pull decompositions and three orderings."""

import csv
from dataclasses import asdict
import sys

from simulate_weight_stages import MODES, Settings, simulate


def main() -> None:
    summaries = ["simulated_seconds", "optimizer_wait_seconds", "publish_wait_seconds",
                 "inline_push_seconds", "final_drain_seconds"]
    writer = csv.DictWriter(sys.stdout, fieldnames=[*asdict(Settings()), "mode", *summaries],
                            lineterminator="\n")
    writer.writeheader()
    for compute in (1, 5):
        for push, pull in ((0, 3), (1, 2), (3, 0)):
            settings = Settings(steps=24, compute_time=compute, push_time=push, pull_time=pull)
            for mode in MODES:
                result = simulate(settings, mode)
                writer.writerow({**asdict(settings), "mode": mode,
                                 **{key: result[key] for key in summaries}})


if __name__ == "__main__":
    main()

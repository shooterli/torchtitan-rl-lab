"""Write the documented blocking/one-sync-overlap CPU comparison as CSV."""

import csv
import sys

from simulate_scheduling import Settings, simulate


def main() -> None:
    fields = ["mode", "sync_time", "target", "simulated_seconds", "groups_per_second",
              "mean_policy_age", "max_policy_age", "batch_wait_seconds",
              "update_wait_seconds", "final_sync_seconds"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for sync_time in (0, 1, 3):
        for target in (0, 1, 3):
            for overlap in (False, True):
                settings = Settings(sync_time=sync_time, overlap_sync=overlap)
                result = simulate(settings, target)
                writer.writerow({"mode": "overlap" if overlap else "blocking",
                                 "sync_time": sync_time,
                                 **{key: result[key] for key in fields
                                    if key not in ("mode", "sync_time")}})


if __name__ == "__main__":
    main()

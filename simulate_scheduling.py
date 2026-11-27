"""Deterministic discrete-event simulation of one-group RL training steps."""

import argparse
from dataclasses import asdict, dataclass
import heapq
import json


@dataclass
class Settings:
    groups: int = 24
    workers: int = 2
    generation_time: int = 2
    train_time: int = 1
    slow_group: int = 0
    slow_multiplier: int = 6
    window_groups: int = 0
    sync_time: int = 0
    overlap_sync: bool = False


def simulate(settings: Settings, target: int) -> dict:
    """Slots remain active through a group's final weight publication."""
    capacity = target + 1
    now = 0
    version = 0
    generator_version = 0
    admitted = 0
    jobs = {}
    generating = []
    training = None
    syncing = None
    last_train_end = 0
    trace = []
    peak_active = 0

    while len(trace) < settings.groups:
        # Process every completion at this timestamp before scheduling new work.
        while generating and generating[0][0] == now:
            _, group_id = heapq.heappop(generating)
            jobs[group_id]["state"] = "ready"
        # Publish the older update before committing a new update at the same time.
        if syncing is not None and syncing["sync_end"] == now:
            generator_version = version
            trace.append(syncing)
            syncing = None
        if training is not None and training["train_end"] == now:
            assert syncing is None, "only one weight synchronization may be in flight"
            version += 1
            last_train_end = training["train_end"]
            syncing = training
            training = None
            if settings.sync_time == 0:
                generator_version = version
                trace.append(syncing)
                syncing = None

        active = len(jobs) + (training is not None) + (syncing is not None)
        while admitted < settings.groups and active < capacity:
            jobs[admitted] = {"group_id": admitted, "state": "waiting"}
            admitted += 1
            active += 1
        peak_active = max(peak_active, active)

        for group_id, job in jobs.items():
            if len(generating) == settings.workers:
                break
            if job["state"] == "waiting":
                duration = settings.generation_time
                if group_id == settings.slow_group:
                    duration *= settings.slow_multiplier
                job.update(state="generating", generation_start=now,
                           generation_end=now + duration, policy_version=generator_version)
                heapq.heappush(generating, (now + duration, group_id))

        if training is None and jobs and (settings.overlap_sync or syncing is None):
            head = min(jobs)
            eligible = [
                group_id for group_id, job in jobs.items()
                if job["state"] == "ready"
                and (settings.window_groups == 0 or group_id < head + settings.window_groups)
            ]
            if eligible:
                training = jobs.pop(min(eligible))
                training.pop("state")
                compute_end = now + settings.train_time
                # The next computation may overlap one sync, but its update waits
                # for that sync to finish before it can start another publication.
                train_end = max(compute_end, syncing["sync_end"] if syncing else 0)
                training.update(train_start=now, compute_end=compute_end,
                                train_end=train_end,
                                sync_end=train_end + settings.sync_time,
                                update_wait=train_end - compute_end,
                                policy_age=version - training["policy_version"],
                                batch_wait=now - last_train_end)

        future = [end for end, _ in generating]
        if training is not None:
            future.append(training["train_end"])
        if syncing is not None:
            future.append(syncing["sync_end"])
        if future:
            now = min(future)
        elif len(trace) < settings.groups:
            raise RuntimeError("scheduler has unfinished work and no future event")

    ages = [row["policy_age"] for row in trace]
    return dict(target=target, capacity=capacity, simulated_seconds=now,
                groups_per_second=settings.groups / now,
                trainer_utilization=settings.groups * settings.train_time / now,
                mean_policy_age=sum(ages) / len(ages), max_policy_age=max(ages),
                batch_wait_seconds=sum(row["batch_wait"] for row in trace),
                update_wait_seconds=sum(row["update_wait"] for row in trace),
                final_sync_seconds=settings.sync_time,
                peak_active=peak_active, trace=trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--generation-time", type=int, default=2)
    parser.add_argument("--train-time", type=int, default=1)
    parser.add_argument("--sync-time", type=int, default=0,
                        help="weight sync after every update, including the last")
    parser.add_argument("--overlap-sync", action="store_true",
                        help="overlap the next computation with one in-flight weight sync")
    parser.add_argument("--slow-group", type=int, default=0, help="-1 disables the straggler")
    parser.add_argument("--slow-multiplier", type=int, default=6)
    parser.add_argument("--window-groups", type=int, default=0, help="0 allows all ready groups")
    parser.add_argument("--targets", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
    if min(settings.groups, settings.workers, settings.generation_time,
           settings.train_time, settings.slow_multiplier) < 1:
        parser.error("counts and durations must be positive")
    if not -1 <= settings.slow_group < settings.groups:
        parser.error("slow-group must be -1 or a valid group id")
    if settings.window_groups < 0 or settings.sync_time < 0 or min(args.targets) < 0:
        parser.error("window-groups, sync-time, and targets must be nonnegative")
    results = [simulate(settings, target) for target in args.targets]
    if args.json:
        print(json.dumps(dict(settings=asdict(settings), results=results), indent=2))
    else:
        print("target  seconds  groups/s  trainer_busy  age_mean  age_max  batch_wait  update_wait")
        for result in results:
            print(f"{result['target']:6}  {result['simulated_seconds']:7}  "
                  f"{result['groups_per_second']:8.3f}  {result['trainer_utilization']:12.1%}  "
                  f"{result['mean_policy_age']:8.3f}  {result['max_policy_age']:7}  "
                  f"{result['batch_wait_seconds']:10}  {result['update_wait_seconds']:11}")


if __name__ == "__main__":
    main()

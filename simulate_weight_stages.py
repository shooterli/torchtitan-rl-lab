"""Idealized weight-publication timelines with always-ready training batches."""

import argparse
from dataclasses import asdict, dataclass
import json


MODES = ("combined", "split", "inline_push")


@dataclass(frozen=True)
class Settings:
    steps: int = 3
    compute_time: int = 1
    optimizer_time: int = 1
    push_time: int = 1
    pull_time: int = 2


def simulate(settings: Settings, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if settings.steps < 1 or min(settings.compute_time, settings.optimizer_time,
                                 settings.push_time, settings.pull_time) < 0:
        raise ValueError("steps must be positive and durations nonnegative")
    start = previous_push_end = previous_pull_end = 0
    trace = []
    for step in range(1, settings.steps + 1):
        compute_end = start + settings.compute_time
        update_barrier = previous_pull_end if mode == "combined" else previous_push_end
        optimizer_start = max(compute_end, update_barrier)
        optimizer_end = optimizer_start + settings.optimizer_time
        publish_start = max(optimizer_end, previous_pull_end)
        push_end = publish_start + settings.push_time
        pull_end = push_end + settings.pull_time
        trace.append(dict(step=step, compute_start=start, compute_end=compute_end,
                          optimizer_start=optimizer_start, optimizer_end=optimizer_end,
                          publish_start=publish_start, push_end=push_end, pull_end=pull_end,
                          optimizer_wait=optimizer_start - compute_end,
                          publish_wait=publish_start - optimizer_end))
        previous_push_end, previous_pull_end = push_end, pull_end
        start = push_end if mode == "inline_push" else publish_start
    return dict(mode=mode, simulated_seconds=previous_pull_end,
                optimizer_wait_seconds=sum(row["optimizer_wait"] for row in trace),
                publish_wait_seconds=sum(row["publish_wait"] for row in trace),
                inline_push_seconds=settings.steps * settings.push_time
                if mode == "inline_push" else 0,
                final_drain_seconds=settings.pull_time
                + (0 if mode == "inline_push" else settings.push_time),
                trace=trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Settings.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=field.default)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = Settings(**{name: getattr(args, name) for name in Settings.__dataclass_fields__})
    try:
        results = [simulate(settings, mode) for mode in args.modes]
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(dict(settings=asdict(settings), results=results), indent=2))
    else:
        print("mode         seconds  optimizer_wait  publish_wait  inline_push  final_drain")
        for row in results:
            print(f"{row['mode']:12} {row['simulated_seconds']:7}  "
                  f"{row['optimizer_wait_seconds']:14}  {row['publish_wait_seconds']:12}  "
                  f"{row['inline_push_seconds']:11}  {row['final_drain_seconds']:11}")


if __name__ == "__main__":
    main()

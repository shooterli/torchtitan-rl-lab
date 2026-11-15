"""Compare cumulative vLLM generation counters from the two smoke-test arms."""

import argparse
import json
from pathlib import Path
from typing import NamedTuple


class Snapshot(NamedTuple):
    start_ns: int
    time_ns: int
    tokens: int


def read_snapshots(run_dir: Path) -> list[Snapshot]:
    files = sorted((run_dir / "vllm_metrics").glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"{run_dir}: expected one generator stream, found {len(files)}")

    path = files[0]
    snapshots = []
    with path.open(encoding="utf-8") as log:
        for line_number, line in enumerate(log, 1):
            try:
                payload = json.loads(line)
                metrics = [
                    metric
                    for resource in payload["resource_metrics"]
                    for scope in resource["scope_metrics"]
                    for metric in scope["metrics"]
                    if metric["name"] == "vllm.generation_tokens"
                ]
                if not metrics:
                    continue
                metric, = metrics
                data = metric["data"]
                if data["aggregation_temporality"] != 2 or data["is_monotonic"] is not True:
                    raise ValueError("expected a monotonic cumulative counter")
                point, = data["data_points"]
                snapshot = Snapshot(
                    point["start_time_unix_nano"], point["time_unix_nano"], point["value"]
                )
                if any(type(value) is not int or value < 0 for value in snapshot):
                    raise ValueError("expected nonnegative integer timestamps and token count")
                if snapshot.start_ns > snapshot.time_ns:
                    raise ValueError("counter starts after the snapshot")
                if snapshots:
                    previous = snapshots[-1]
                    if snapshot.start_ns != previous.start_ns or snapshot.tokens < previous.tokens:
                        raise ValueError("generation counter reset")
                    if snapshot.time_ns <= previous.time_ns:
                        raise ValueError("snapshot timestamps must increase")
                snapshots.append(snapshot)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="parent directory for the two runs")
    parser.add_argument("--fixed-cap", action="store_true", help="read the fixed-cap runs")
    parser.add_argument("--skip-snapshots", type=int, required=True)
    parser.add_argument("--intervals", type=int, required=True)
    args = parser.parse_args()
    if args.skip_snapshots < 0 or args.intervals < 1:
        parser.error("skip-snapshots must be nonnegative and intervals must be positive")

    results = []
    suffix = "fixed_cap" if args.fixed_cap else "smoke"
    for arm in (f"sync_{suffix}", f"async_{suffix}"):
        try:
            snapshots = read_snapshots(args.run_dir / arm)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        end_index = args.skip_snapshots + args.intervals
        if end_index >= len(snapshots):
            parser.error(f"{arm}: need {end_index + 1} counter snapshots, found {len(snapshots)}")
        start, end = snapshots[args.skip_snapshots], snapshots[end_index]
        seconds = (end.time_ns - start.time_ns) / 1e9
        tokens = end.tokens - start.tokens
        results.append(f"{arm}: {tokens:,} generated tokens / {seconds:.3f} s = {tokens / seconds:,.2f} tokens/s")
    print("\n".join(results))


if __name__ == "__main__":
    main()

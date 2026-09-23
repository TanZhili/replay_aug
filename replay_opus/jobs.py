"""Batch execution shared by the commands: a job list, a worker pool, one manifest."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from .config import Config


def save_config(config: Config, out_root: Path) -> None:
    """Keep the effective config with the outputs: scenes can be rebuilt from it."""
    text = yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False)
    (Path(out_root) / "config.yaml").write_text(text, encoding="utf-8")


def run_jobs(
    jobs: list,
    worker: Callable[[tuple], list[dict]],
    initializer: Callable,
    initargs: tuple,
    workers: int,
    manifest_path: Path,
    describe: Callable[[tuple], str] = lambda job: repr(job[:2]),
) -> tuple[int, int]:
    """Run ``worker`` over ``jobs`` and write every returned record to the manifest.

    A worker returns a list of records; a record holding ``error`` counts as a
    failure. A job that raises becomes one error record, so one bad input never stops
    the batch. Returns ``(written, failed)``.
    """
    written = failed = 0
    total = len(jobs)

    def record_all(records: list[dict], manifest) -> None:
        nonlocal written, failed
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            if "error" in record:
                failed += 1
                print(
                    f"  FAILED {record.get('input', record.get('job'))}: {record['error']}",
                    file=sys.stderr,
                )
            else:
                written += 1

    def safe(job) -> list[dict]:
        try:
            return worker(job)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            return [{"job": describe(job), "error": repr(exc)}]

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        if workers == 1:
            initializer(*initargs)
            for done, job in enumerate(jobs, start=1):
                record_all(safe(job), manifest)
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}", end="\r", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=initializer, initargs=initargs
            ) as pool:
                futures = {pool.submit(worker, job): job for job in jobs}
                for done, future in enumerate(as_completed(futures), start=1):
                    try:
                        records = future.result()
                    except Exception as exc:  # noqa: BLE001
                        records = [{"job": describe(futures[future]), "error": repr(exc)}]
                    record_all(records, manifest)
                    if done % 10 == 0 or done == total:
                        print(f"  {done}/{total}", end="\r", flush=True)
    print()
    return written, failed


__all__ = ["run_jobs", "save_config"]

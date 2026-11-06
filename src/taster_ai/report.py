"""`taster report`: what the logs in .taster/logs say happened."""

from __future__ import annotations

import glob
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


def read_records(folder: str, days: float | None = None) -> list[dict[str, Any]]:
    """Every record in a log folder, oldest first, from the last `days` days."""
    cutoff = None if days is None else datetime.now(timezone.utc) - timedelta(days=days)
    records = []
    for path in glob.glob(os.path.join(folder, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if cutoff is None or datetime.fromisoformat(record["ts"]) >= cutoff:
                    records.append(record)
    return sorted(records, key=lambda r: r.get("ts", ""))


def summary(records: Iterable[dict[str, Any]], sample_rules: bool = False) -> str:
    records = list(records)
    checks = [r for r in records if "check" in r]
    events = Counter(r["event"] for r in records if "event" in r)
    would_block = [r for r in checks if r.get("would_action") == "blocked"]
    # Judged as breaking the rules, but below the target's threshold: enforce
    # mode would let these run, so a threshold set too high shows up here.
    unsure = [r for r in checks if r.get("label") == "breaks" and r.get("would_action") != "blocked"]
    latencies = [r["latency_s"] for r in checks if r.get("label") != "error" and "latency_s" in r]
    rules = {r["rules"] for r in records if r.get("rules")}

    lines = [
        f"Calls judged       {len(checks)}",
        f"Would be blocked   {len(would_block)}",
        f"Blocked            {sum(1 for r in checks if r.get('action') == 'blocked')}",
        f"Breaks, low conf.  {len(unsure)}",
        f"Unclear            {sum(1 for r in checks if r.get('label') == 'unclear')}",
        f"Judge failures     {sum(1 for r in checks if r.get('label') == 'error')}",
    ]
    if latencies:
        lines.append(f"Median check time  {statistics.median(latencies):.1f} s")
    if would_block:
        lines.append("\nWould be blocked:")
        for r in would_block[-20:]:
            why = (r.get("evidence") or [""])[0]
            lines.append(f"  {r.get('ts', '')}  {r.get('tool')}  {why}")

    if unsure:
        lines.append("\nBreaks the rules, but below the threshold, so enforce would let these run:")
        for r in unsure[-20:]:
            why = (r.get("evidence") or [""])[0]
            lines.append(f"  {r.get('ts', '')}  {r.get('tool')}  {r.get('confidence')}  {why}")

    warnings = []
    if sample_rules:
        warnings.append("the rules are still the samples, so these results don't reflect your rules")
    if not checks:
        warnings.append("no calls were judged: check your scope in .taster/taster.yaml")
    if len(rules) > 1:
        warnings.append(f"the rules or scope changed {len(rules) - 1} time(s) in this period")
    if events["config-error"]:
        warnings.append(f"{events['config-error']} config error(s): the last good rules stayed in use")
    if events["dropped"]:
        warnings.append(f"{events['dropped']} call(s) not judged: too many were waiting")
    if events["service-start"] > events["service-stop"] + 1:
        warnings.append("the service stopped unexpectedly: calls may have gone unchecked meanwhile")
    if warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  {w}" for w in warnings)
    return "\n".join(lines)

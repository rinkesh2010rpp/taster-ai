"""The `taster` command.

    taster init claude-code     set up this project
    taster report [--days 14]   see what happened
    taster report --export F    write the records to a file to share
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _init(args: argparse.Namespace) -> int:
    from .adapters.claude_code.install import init

    return init(os.getcwd())


def _report(args: argparse.Namespace) -> int:
    from .adapters.claude_code.install import template
    from .report import read_records, summary

    folder = os.path.join(os.getcwd(), ".taster")
    records = read_records(os.path.join(folder, "logs"), args.days)
    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} records to {args.export}. Read it before sharing:")
        print("records hold tool names, verdicts and the judge's reasons, and a reason can quote a call.")
        return 0
    sample = False
    try:
        with open(os.path.join(folder, "constitution.md"), encoding="utf-8") as f:
            sample = f.read() == template("constitution.md")
    except OSError:
        pass
    print(summary(records, sample_rules=sample))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taster", description="Checks on what your AI agent does.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="set up this project")
    init.add_argument("agent", choices=["claude-code"])
    init.set_defaults(run=_init)
    report = commands.add_parser("report", help="see what happened")
    report.add_argument("--days", type=float, default=14.0, help="how far back to look (default 14)")
    report.add_argument("--export", metavar="FILE", help="write the records to a file")
    report.set_defaults(run=_report)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

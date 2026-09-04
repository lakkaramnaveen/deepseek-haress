#!/usr/bin/env python3
"""DeepSeek coding-model test harness.

Usage:
    python cli.py run                # run every task in tasks/
    python cli.py run --task fizzbuzz  # run a single task by id
    python cli.py list               # list available tasks
"""

import argparse
import sys

from dotenv import load_dotenv

from harness.client import DeepSeekClient
from harness.runner import load_all_tasks, run_all


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DeepSeek coding-model test harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run task(s) against DeepSeek and score in sandbox")
    run_parser.add_argument("--task", help="Run only the task with this id", default=None)

    subparsers.add_parser("list", help="List available tasks")

    args = parser.parse_args()

    if args.command == "list":
        tasks = load_all_tasks()
        if not tasks:
            print("No tasks found in tasks/")
            return
        for t in tasks:
            print(f"  {t['id']:<20} [{t['language']}]")
        return

    if args.command == "run":
        try:
            client = DeepSeekClient()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            results = run_all(client, task_filter=args.task)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        print(f"\n{passed}/{total} tasks passed")
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

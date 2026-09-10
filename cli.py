#!/usr/bin/env python3
"""DeepSeek coding-model test harness.

Usage:
    python cli.py run                # run every task in tasks/
    python cli.py run --task fizzbuzz  # run a single task by id
    python cli.py list               # list available tasks
    python cli.py translate --src ./myapp --out ./myapp-js \\
        --from python --to javascript
"""

import argparse
import sys

from dotenv import load_dotenv

from harness.client import DeepSeekClient
from harness.runner import load_all_tasks, run_all
from harness.translate_agent import LANGUAGE_EXTENSIONS, translate_codebase, verify_codebase


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DeepSeek coding-model test harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run task(s) against DeepSeek and score in sandbox")
    run_parser.add_argument("--task", help="Run only the task with this id", default=None)

    subparsers.add_parser("list", help="List available tasks")

    tr_parser = subparsers.add_parser(
        "translate", help="Translate an entire codebase from one language to another"
    )
    tr_parser.add_argument("--src", required=True, help="Source codebase directory")
    tr_parser.add_argument("--out", required=True, help="Output directory for translated code")
    tr_parser.add_argument("--from", dest="source_language", required=True, choices=sorted(LANGUAGE_EXTENSIONS))
    tr_parser.add_argument("--to", dest="target_language", required=True, choices=sorted(LANGUAGE_EXTENSIONS))
    tr_parser.add_argument("--no-copy-other", action="store_true", help="Don't carry over non-source files verbatim")
    tr_parser.add_argument("--verify", action="store_true", help="Run the translated project in the sandbox afterwards")
    tr_parser.add_argument("--run", dest="run_cmd", help="Command to run inside the sandbox for --verify, e.g. 'npm test'")
    tr_parser.add_argument("--install", dest="install_cmd", default=None, help="Command to install deps before --run, e.g. 'npm install'")
    tr_parser.add_argument("--network", action="store_true", help="Allow network access during --verify (needed for --install)")
    tr_parser.add_argument("--timeout", type=int, default=180, help="Sandbox timeout in seconds for --verify")

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

    if args.command == "translate":
        if args.verify and not args.run_cmd:
            print("Error: --verify requires --run", file=sys.stderr)
            sys.exit(1)

        try:
            client = DeepSeekClient()
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        def on_progress(i, total, rel_path):
            print(f"  [{i}/{total}] translating {rel_path} ...")

        print(
            f"Translating {args.src} ({args.source_language}) -> "
            f"{args.out} ({args.target_language})"
        )
        try:
            report = translate_codebase(
                client,
                src_dir=args.src,
                out_dir=args.out,
                source_language=args.source_language,
                target_language=args.target_language,
                copy_other_files=not args.no_copy_other,
                on_progress=on_progress,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{report.succeeded}/{len(report.files)} files translated")
        if report.other_files_copied:
            print(f"{report.other_files_copied} other files copied as-is")
        for f in report.files:
            if not f.ok:
                print(f"  FAILED {f.source_path}: {f.error}")

        if report.failed:
            sys.exit(1)

        if args.verify:
            print(f"\nVerifying translated project in sandbox: {args.run_cmd!r}")
            result = verify_codebase(
                language=args.target_language,
                project_dir=args.out,
                run_cmd=args.run_cmd,
                install_cmd=args.install_cmd,
                timeout=args.timeout,
                network=args.network,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if result.error:
                print(f"Error: {result.error}", file=sys.stderr)
            print("PASS" if result.passed else "FAIL")
            sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()

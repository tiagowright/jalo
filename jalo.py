#!/usr/bin/env python
"""Command-line REPL entry point for the Jalo keyboard tooling."""

from __future__ import annotations

import argparse
import sys

from repl.shell import JaloShell, _configure_readline, DEFAULT_CONFIG_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jalo is just another keyboard layout optimizer.")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable readline history persistence.",
    )
    args = parser.parse_args(argv)

    if not args.no_history:
        _configure_readline()

    try:
        shell = JaloShell(config_path=DEFAULT_CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - surface instantiation issues
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        shell._info("")
        shell._info("Interrupted. Bye bye.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

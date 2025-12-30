#!/usr/bin/env python
"""Command-line REPL entry point for the Jalo keyboard tooling."""

from __future__ import annotations

import argparse
import sys

from repl.shell import JaloShell, _configure_readline, DEFAULT_CONFIG_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Jalo is just another keyboard layout optimizer. Run with no arguments to enter interactive mode, or provide a series of commands or a script file to execute.",
        epilog="Jalo can analyze, generate, optimize, edit keyboard layouts. Try `jalo.py` to go into interactive mode, or from the command line `jalo.py -c help` to learn more about the commands."
    )
    parser.add_argument(
        "script_file",
        nargs = '?',
        default=None,
        help="Optional script file containing jalo commands, one per line. Executes then exits."
    )
    parser.add_argument(
        "-c",
        "--command",
        action="append",
        help="Command to execute. Can be specified multiple times to execute a sequence of commands, then exit."
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable readline history persistence.",
    )
    args = parser.parse_args(argv)

    if args.command and args.script_file:
        print("Error: Cannot specify both a script file and commands.", file=sys.stderr)
        return 1
    
    script_str = ""
    if args.command:
        script_str = "\n".join(args.command)

    if args.script_file:
        args.no_history = True
        try:
            with open(args.script_file, 'r') as file:
                script_str = file.read()
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if not args.no_history:
        _configure_readline()

    try:
        shell = JaloShell(config_path=DEFAULT_CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - surface instantiation issues
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not script_str:
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            shell._info("")
            shell._info("Interrupted. Bye bye.\n")
    else:
        try:
            shell.script(script_str)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

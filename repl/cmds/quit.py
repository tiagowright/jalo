"""`quit` command for the Jalo shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repl.shell import Command

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="quit",
        description="Quits Jalo.",
        arguments=(),
        examples=("",),
        category="commands",
        short_description="quits Jalo",
    )


def exec(shell: "JaloShell", arg: str) -> bool:
    shell._info("Exiting Jalo REPL. Bye bye.")
    shell._info("")
    return True


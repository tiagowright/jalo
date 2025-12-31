"""`exit` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.shell import Command
from . import quit as quit_cmd

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="exit",
        description="Quits Jalo.",
        arguments=(),
        examples=("",),
        category="commands",
        short_description="quits Jalo",
    )


def exec(shell: "JaloShell", arg: str) -> bool:
    return quit_cmd.exec(shell, arg)


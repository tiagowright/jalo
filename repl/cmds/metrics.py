"""`metrics` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.formatting import format_metrics_table
from repl.shell import Command

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="metrics",
        description="Shows all current metrics and their descriptions.",
        arguments=(),
        examples=("",),
        category="analysis",
        short_description="shows metric names and descriptions",
    )


def exec(shell: "JaloShell", arg: str) -> None:
    shell._info(format_metrics_table(shell.metrics, set(shell.break_before_metrics)))


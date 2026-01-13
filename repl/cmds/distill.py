"""`distill` command for the Jalo shell."""

from distill import Distillator
from typing import TYPE_CHECKING

from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="distill",
        description="Attempt to distill an objective function from a target layout.",
        arguments=(
            CommandArgument(
                "target",
                "<target>",
                "the keyboard layout to distill an objective function from, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0", "graphite"),
        category="optimization",
        short_description="distill an objective function from a target layout",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return list_keyboard_names(shell, text)


def exec(shell: "JaloShell", arg: str) -> None:
    layouts = parse_keyboard_names(shell, arg)
    if layouts is None or len(layouts) > 1:
        shell._warn("specify a single keyboard name to distill.")
        return

    distillator = Distillator(shell.model)

    objective, char_at_pos_scores = distillator.distill(layouts[0])
    shell._info(f"Distilled the following:\nobjective {objective}\n")
    
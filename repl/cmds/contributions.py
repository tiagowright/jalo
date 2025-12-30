"""`contributions` command for the Jalo shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="contributions",
        description=(
            "Shows what is driving the score of one or more keyboard layouts, by tabulating the contributions that every metric makes to the total score."
        ),
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard> [<keyboard>...]",
                "the keyboard layouts to analyze, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0 1", "0 sturdy qwerty graphite hdpm"),
        category="analysis",
        short_description="to understand the score, tabulates the contributions of each metric",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return shell._list_keyboard_names(text)


def exec(shell: "JaloShell", arg: str) -> None:
    layouts = shell._parse_keyboard_names(arg)
    if layouts is None:
        shell._warn("usage: contributions <keyboard> [<keyboard>...]")
        return

    shell._info(shell._tabulate_analysis(layouts, show_contributions=True))


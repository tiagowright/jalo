"""`analyze` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="analyze",
        description="Analyze the given keyboard layout, prints the layout, hardware, all metrics, and score.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to analyze, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0", "qwerty"),
        category="analysis",
        short_description="analyze a keyboard layout on all metrics",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return list_keyboard_names(shell, text)


def exec(shell: "JaloShell", arg: str) -> None:
    layouts = parse_keyboard_names(shell, arg)
    if layouts is None or len(layouts) > 1:
        shell._warn("specify a single keyboard name to analyze.")
        return

    shell.do_show(arg)  # pyright: ignore[reportAttributeAccessIssue]
    shell._info(shell._tabulate_analysis(layouts))

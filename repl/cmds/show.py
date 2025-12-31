"""`show` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.formatting import format_layout_display
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="show",
        description="Shows the given keyboard layout, prints the layout and hardware.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to show, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0", "qwerty", "1.1"),
        category="analysis",
        short_description="show a keyboard layout",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return list_keyboard_names(shell, text)


def exec(shell: "JaloShell", arg: str) -> None:
    layouts = parse_keyboard_names(shell, arg)
    if layouts is None or len(layouts) != 1:
        shell._warn("usage: show <keyboard>")
        return

    shell._info("")
    shell._info(format_layout_display(layouts[0]))
    shell._info("")

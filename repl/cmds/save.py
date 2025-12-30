"""`save` command for the Jalo shell."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="save",
        description="Saves a layout to the ./layouts directory. Saved layouts can then be used in other commands by name.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to save, can be a layout name or the index of a layout in memory.",
            ),
            CommandArgument("name", "[<name>]", "the name of the new layout, defaults to the home row characters."),
        ),
        examples=("0", "1 mylayout"),
        category="editing",
        short_description="save new layouts from memory to a new file",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return shell._list_keyboard_names(text)


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)
    layouts = shell._parse_keyboard_names(args[0]) if args else None
    if layouts is None or len(layouts) != 1:
        shell._warn("usage: save <keyboard> [<name>]")
        return

    layout = layouts[0]

    if len(args) > 1:
        name_candidate = args[1]
    else:
        name_candidate = "".join(key.char for key in layout.keys if key.position.is_home)

    filename = f"{name_candidate}.kb"
    filepath = os.path.join("layouts", filename)

    if os.path.exists(filepath):
        shell._warn(
            f"layout file already exists: {filepath}, not overwriting. Specify a different name: save <keyboard> <name>"
        )
        return

    with open(filepath, "w", encoding="utf-8") as file_handle:
        file_handle.write(str(layout))

    shell._info(f"saved layout to {filepath}")


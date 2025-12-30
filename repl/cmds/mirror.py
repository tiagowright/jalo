"""`mirror` command for the Jalo shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="mirror",
        description="Mirrors a keyboard layout horizontally, so that the left and right hands are swapped.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to mirror, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0", "hdpm"),
        category="editing",
        short_description="mirrors a keyboard layout horizontally",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return shell._list_keyboard_names(text)


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if len(args) != 1:
        shell._warn("usage: mirror <keyboard>")
        return

    layouts = shell._parse_keyboard_names(args[0])
    if layouts is None or len(layouts) != 1:
        shell._warn("usage: mirror <keyboard>")
        return

    layout = layouts[0]
    mirrored_layout = layout.mirror()

    shell._push_layout_to_stack(mirrored_layout, layout)

    shell._info("")
    shell._info(shell._layout_memory_to_str(list_num=0, top_n=1))


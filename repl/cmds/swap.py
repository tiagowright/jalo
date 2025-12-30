"""`swap` command for the Jalo shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="swap",
        description="Swaps position pairs on the keyboard, for those hand crafted, fine tuning adjustments to a layout.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to swap positions on, can be a layout name or the index of a layout in memory.",
            ),
            CommandArgument("pair", "<pair> [<pair> ...]", "the character pairs to swap positions"),
        ),
        examples=("0 dh", "qwerty aq wz"),
        category="editing",
        short_description="swaps two or more positions on the keyboard",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    arg_num = shell._arg_num_at_index(line, begidx, endidx)
    if arg_num is None or arg_num <= 1:
        return shell._list_keyboard_names(text)

    return [char for char in shell.model.freqdist.char_seq if char != shell.model.freqdist.out_of_distribution]


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if len(args) < 2:
        shell._warn("usage: swap <keyboard> <pair> [<pair> ...]")
        return

    layouts = shell._parse_keyboard_names(args[0])
    if layouts is None or len(layouts) != 1:
        shell._warn("usage: swap <keyboard> <pair> [<pair> ...]")
        return

    layout = layouts[0]

    pairs = args[1:]
    for pair in pairs:
        if len(pair) != 2:
            shell._warn(f"each pair must be two characters, could not understand pair: {pair}")
            return

        char1, char2 = pair
        if char1 not in layout.char_to_key:
            shell._warn(f"character {char1} not found in layout")
            return
        if char2 not in layout.char_to_key:
            shell._warn(f"character {char2} not found in layout")
            return

    swapped_layout = layout.swap(char_pairs=[(char1, char2) for char1, char2 in pairs])

    shell._push_layout_to_stack(swapped_layout, layout)

    shell._info("")
    shell._info(shell._layout_memory_to_str(list_num=0, top_n=1))


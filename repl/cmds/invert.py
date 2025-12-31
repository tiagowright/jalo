"""`invert` command for the Jalo shell."""



from typing import TYPE_CHECKING

from hardware import Hand
from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="invert",
        description="Inverts top and bottom rows of a keyboard layout (mirrors vertically).",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to invert, can be a layout name or the index of a layout in memory.",
            ),
            CommandArgument("hand", "[hand=both]", "`left` or `right` to invert the left or right hand, default is `both`."),
        ),
        examples=("0", "hdpm right"),
        category="editing",
        short_description="inverts top and bottom rows of a keyboard layout (mirrors vertically)",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    arg_num = shell._arg_num_at_index(line, begidx, endidx)
    if arg_num is None or arg_num <= 1:
        return list_keyboard_names(shell, text)

    return [word for word in ("left", "right", "both") if word.startswith(text)]


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if len(args) < 1:
        shell._warn("specify a keyboard. Usage: invert <keyboard> [hand=both]")
        return

    if len(args) > 2:
        shell._warn("too many arguments. Usage: invert <keyboard> [hand=both]")
        return

    if len(args) >= 2:
        hand_str = args[1].lower()
        if hand_str not in ("left", "right", "both"):
            shell._warn("invalid hand: {hand_str}, specify `left` or `right` or `both`. Usage: invert <keyboard> [hand=both]")
            return
    else:
        hand_str = "both"

    hand = None if hand_str == "both" else Hand.LEFT if hand_str == "left" else Hand.RIGHT

    layouts = parse_keyboard_names(shell, args[0])
    if layouts is None or len(layouts) != 1:
        shell._warn("invalid keyboard: {args[0]}. Usage: invert <keyboard> [hand=both]")
        return

    layout = layouts[0]
    inverted_layout = layout.invert(hand=hand)

    shell._push_layout_to_stack(inverted_layout, layout)

    shell._info("")
    shell._info(shell._layout_memory_to_str(list_num=0, top_n=1))

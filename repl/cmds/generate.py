"""`generate` command for the Jalo shell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optim import Optimizer
from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="generate",
        description=(
            "Generates new layouts from scratch, starting from a number of random seeds and improving them to find the best (lowest) score, "
            "based on the current objective (see `help objective`). "
            "If a keyboard layout is provided, it will be used to determine the hardware and the location of pinned characters (see `help pin`). "
            "If no keyboard layout is provided, default hardware is used and pinned characters are ignored with a warning."
        ),
        arguments=(
            CommandArgument(
                "seeds",
                "[seeds=100]",
                "the number of random seeds to generate, default is 100, which is enough for most cases.",
            ),
            CommandArgument(
                "keyboard",
                "[<keyboard>]",
                "the keyboard layout to improve, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("", "1000", "100 hdpm"),
        category="optimization",
        short_description="generates a wide variety of new layouts from scratch",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    arg_num = shell._arg_num_at_index(line, begidx, endidx)
    if arg_num is None or arg_num <= 1:
        seed_suggestions = ["10", "20", "100", "1000"]
        return [suggestion for suggestion in seed_suggestions if suggestion.startswith(text)]
    return list_keyboard_names(shell, text)


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    seeds = 100
    layout = None

    if len(args) > 0:
        try:
            seeds = int(args[0])
        except ValueError:
            shell._warn("seeds must be an integer: generate [seeds=100] [keyboard]")
            return

    if len(args) > 1:
        layouts = parse_keyboard_names(shell, args[1])
        if layouts is None or len(layouts) != 1:
            shell._warn("usage: generate <seeds> [keyboard]")
            return
        layout = layouts[0]

    if layout is not None:
        model = shell._get_model(layout)
        pinned_positions = model.pinned_positions_from_layout(layout, shell.pinned_chars)
    else:
        model = shell.model
        pinned_positions = ()
        if shell.pinned_chars:
            pins = ", ".join(shell.pinned_chars)
            shell._warn(
                "Warning: no keyboard layout provided, cannot pin characters to a position, "
                f"pins will be ignored ({pins})"
            )

    shell._info(f"generating {seeds} seeds.")

    optimizer = Optimizer(model, population_size=100, solver="genetic")

    if layout is not None:
        optimizer.generate(
            seeds=seeds,
            char_at_pos=tuple(model.char_at_positions_from_layout(layout)),
            pinned_positions=pinned_positions,
        )
    else:
        n_positions = len(model.hardware.positions)
        char_seq = shell.freqdist.char_seq[:n_positions]
        optimizer.generate(char_seq=char_seq, seeds=seeds, pinned_positions=pinned_positions)

    list_num = shell._layout_memory_from_optimizer(optimizer, push_to_stack=False)

    shell._info("")
    shell._info(shell._layout_memory_to_str(list_num=list_num))

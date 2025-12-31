"""`polish` command for the Jalo shell."""



import shlex
from typing import TYPE_CHECKING

from optim import Optimizer
from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.formatting import format_table
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="polish",
        description="Identifies small number of swaps that can improve the score of a given layout.",
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to polish, can be a layout name or the index of a layout in memory.",
            ),
        ),
        examples=("0", "qwerty"),
        category="optimization",
        short_description="identifies small number of swaps that can improve the score of a given layout",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    return list_keyboard_names(shell, text)


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)
    layouts = parse_keyboard_names(shell, args[0]) if args else None

    if layouts is None or len(layouts) != 1:
        shell._warn("usage: polish <keyboard>")
        return

    layout = layouts[0]
    model = shell._get_model(layout)
    char_at_pos = tuple(model.char_at_positions_from_layout(layout))
    pinned_positions = model.pinned_positions_from_layout(layout, shell.pinned_chars)

    optimizer = Optimizer(model, population_size=100, solver="annealing")
    swaps_scores = optimizer.polish(
        char_at_pos=char_at_pos,
        iterations=100,
        max_depth=3,
        pinned_positions=pinned_positions,
    )

    by_len: dict[int, list[tuple[tuple[tuple[int, int], ...], float]]] = {}
    for swaps, score in swaps_scores.items():
        len_swaps = len(swaps)
        if len_swaps == 0:
            continue
        by_len.setdefault(len_swaps, []).append((swaps, score))

    parser = shlex.shlex()

    for n_swaps in sorted(by_len.keys()):
        shell._info(f"{n_swaps} swaps:")
        rows = []
        sorted_swaps_scores = sorted(by_len[n_swaps], key=lambda x: x[1])
        for swaps, score in sorted_swaps_scores[:10]:
            swapped_char_at_pos = list(char_at_pos)
            chars = []
            for i, j in swaps:
                chari = model.freqdist.char_seq[swapped_char_at_pos[i]]
                charj = model.freqdist.char_seq[swapped_char_at_pos[j]]
                if chari in parser.quotes or chari in parser.escape:
                    chari = f"\\{chari}"
                if charj in parser.quotes or charj in parser.escape:
                    charj = f"\\{charj}"
                chars.append(f"{chari}{charj}")
                swapped_char_at_pos[i], swapped_char_at_pos[j] = swapped_char_at_pos[j], swapped_char_at_pos[i]

            chars_str = " ".join(chars)
            rows.append([f"swap {layout.name} {chars_str}", f"# score: {100 * score:.2f}"])
        shell._info(format_table(rows))
        shell._info("")

    shell._info("")

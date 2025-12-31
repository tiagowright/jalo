"""`improve` command for the Jalo shell."""



from typing import TYPE_CHECKING

from optim import Optimizer
from repl.completers import list_keyboard_names, parse_keyboard_names
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="improve",
        description=(
            "Tries to improve the score of a given layout by swapping positions and columns. "
            "Produces layouts that are not very different from the original layout, but that score better if possible. "
            "While `generate` produces a wide variety of very different layouts that are a minimum distance from each other, "
            "`improve` does the opposite, producing layouts that are within a maximum distance. So these are complementary steps, "
            "with `generate` providing large coarse grained cuts, and `improve` refining it further, and finally `polish` helping with the final "
            "few swaps."
        ),
        arguments=(
            CommandArgument(
                "keyboard",
                "<keyboard>",
                "the keyboard layout to improve, can be a layout name or the index of a layout in memory.",
            ),
            CommandArgument(
                "seeds",
                "[seeds=100]",
                "the number of random seeds to generate, default is 100, which is enough for most cases.",
            ),
        ),
        examples=("0", "qwerty", "hdpm 200"),
        category="optimization",
        short_description="try to improve the score of a given layout (neighboring layouts)",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    arg_num = shell._arg_num_at_index(line, begidx, endidx)
    if arg_num is None or arg_num <= 1:
        return list_keyboard_names(shell, text)

    seed_suggestions = ["10", "20", "100", "200", "1000"]
    return [suggestion for suggestion in seed_suggestions if suggestion.startswith(text)]


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if not args:
        shell._warn("usage: improve <keyboard> [seeds=100]")
        return

    layouts = parse_keyboard_names(shell, args[0])
    if layouts is None or len(layouts) != 1:
        shell._warn("usage: improve <keyboard>")
        return

    if len(args) > 1:
        try:
            seeds = int(args[1])
        except ValueError:
            shell._warn("iterations must be an integer: improve <keyboard> [seeds=100]")
            return
    else:
        seeds = 100

    layout = layouts[0]
    shell._info(f"improving {layout.name} with {seeds} seeds.")

    model = shell._get_model(layout)
    pinned_positions = model.pinned_positions_from_layout(layout, shell.pinned_chars)

    optimizer = Optimizer(model, population_size=100, solver="annealing")
    optimizer.improve(
        char_at_pos=tuple(model.char_at_positions_from_layout(layout)),
        seeds=seeds,
        hamming_distance_threshold=10,
        pinned_positions=pinned_positions,
    )

    list_num = shell._layout_memory_from_optimizer(optimizer, original_layout=layout, push_to_stack=False)

    shell._info("")
    shell._info(shell._layout_memory_to_str(list_num=list_num))

"""`pin` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="pin",
        description=(
            "Pins characters to their current positions, so that they cannot be swapped or moved to a different position during editing (`generate`, `improve`, etc.). "
            "To see current pins, pass no arguments (`pin`). To clear all pins, use `pin nothing`."
        ),
        arguments=(
            CommandArgument("`nothing`", "[nothing]", "to remove all existing pins, specify `pin nothing`."),
            CommandArgument(
                "chars",
                "[<chars> ...]",
                "the characters to pin, can be a single, multiple characters, or a space-separated list of characters.",
            ),
        ),
        examples=("", "a b c", "asdf", "aeiou", "nothing"),
        category="optimization",
        short_description="pins characters to their current position",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    words = [word for word in ("nothing", "clear") if word.startswith(text)]

    if words and endidx - begidx >= 2:
        return words

    return words + [char for char in shell.model.freqdist.char_seq if char != shell.model.freqdist.out_of_distribution]


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if not args:
        pass
    elif args[0].lower() in ("nothing", "clear"):
        shell.pinned_chars = []
    else:
        all_chars = [char for chars in args for char in chars if char not in shell.pinned_chars]

        invalid_chars = [char for char in all_chars if char not in shell.model.freqdist.char_seq]
        if invalid_chars:
            str_invalid_chars = " ".join(invalid_chars)
            shell._warn(f"Warning: {len(invalid_chars)} character(s) are not in the corpus: {str_invalid_chars}")

        shell.pinned_chars.extend(all_chars)

    if not shell.pinned_chars:
        shell._info("nothing pinned.")
    else:
        str_pinned_chars = " ".join(shell.pinned_chars)
        shell._info(f"pinned: {str_pinned_chars}")
    shell._info("")


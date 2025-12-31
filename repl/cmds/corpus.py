"""`corpus` command for the Jalo shell."""



import re
from pathlib import Path
from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument
from freqdist import FreqDist

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="corpus",
        description="Shows or changes the text corpus used to score layouts. "
                "The corpus determines the frequency of characters, bigrams, trigrams, skipgrams that are used to calculate each metric and compute the score. "
                "If no name is specified, shows the current corpus.",
        arguments=(
            CommandArgument(
                "name",
                "[<name>]",
                "when specified, loads the corpus from `./corpus/<name>/`. If not specified, shows the current corpus.",
            ),
        ),
        examples=("", "en", "shai"),
        category="analysis",
        short_description="view or update the text corpus used to score layouts",
    )



def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    # list every directory in ./corpus/ that starts with the text and that contains a file called monograms.json
    return [f.parent.stem for f in Path("corpus").glob(f"{text}*/monograms.json")]


def exec(shell: "JaloShell", arg: str) -> None:
    if not arg:
        shell._info(f"Current corpus: {shell.freqdist.corpus_name}")
        return

    try:
        new_corpus = FreqDist.from_name(arg)
    except ValueError as e:
        shell._warn(f"error loading corpus: {e}")
        return

    shell._change_settings(freqdist=new_corpus)
    shell._info(f"Updated corpus to: {new_corpus.corpus_name}")


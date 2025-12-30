"""`objective` command for the Jalo shell."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from objective import ObjectiveFunction
from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="objective",
        description="\n".join(
            [
                "Shows the current objective function when called without an argument, or sets the objective function "
                "from a configuration file or to a new explicit formula. "
                "The objective function is used to score keyboard layouts, and it is considered a measure of effort, meaning lower scores are better. "
                "Optimization commands (e.g., `generate`, `improve`, `polish`) use the objective function's score to find the best layouts possible.",
                "",
                "Objective functions can be loaded from a toml file in ./objectives/ by naming the file, or can be hand crafted directly by typing a formula.",
                "",
                "The formula is any linear combination of metrics:"
                "  [+|-][weight_1]<metric_name_1> [+|- [weight_2]<metric_name_2> ...]",
                "",
                "where:",
                "  <weight_i> is a float.",
                "  <metric_name_i> is the name of a valid metric (type `metrics`)",
                "",
                "Note that there is no `*` between the weight and the metric name, but a space is allowed.",
            ]
        ),
        arguments=(
            CommandArgument(
                "formula",
                "[<name> | <formula>]",
                "when specified, sets the objective function to the given formula or file.",
            ),
        ),
        examples=("", "default", "oxeylyzer", "100sfb + 6effort + 60pinky_ring + 60scissors_ortho + 60sfs - 5alt"),
        category="optimization",
        short_description="view or update the objective function used to score layouts",
    )


_SIGN_PATTERN = re.compile(r"[+-]")
_FLOAT_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    i = 0
    match = _SIGN_PATTERN.match(text, i)
    if match:
        i = match.end()

    match = _FLOAT_PATTERN.match(text, i)
    if match:
        i = match.end()

    ret: list[str] = []

    if i == 0:
        arg_num = shell._arg_num_at_index(line, begidx, endidx)
        if arg_num is None or arg_num <= 1:
            ret.extend([f.stem for f in Path("objectives").glob(f"{text}*.toml")])

    ret.extend([text[:i] + metric.name for metric in shell.metrics if metric.name.startswith(text[i:])])

    return ret


def exec(shell: "JaloShell", arg: str) -> None:
    if not arg:
        shell._info(f"Current function:\nobjective {shell.objective}\n")
        return

    try:
        new_objective = ObjectiveFunction.from_formula(arg)
    except ValueError as e:
        try:
            new_objective = ObjectiveFunction.from_name(arg)
        except ValueError:
            shell._warn(f"error parsing objective function: {e}")
            return

    shell._load_objective(new_objective)
    shell._info(f"Updated function to:\nobjective {shell.objective}\n")


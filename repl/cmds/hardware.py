"""`hardware` command for the Jalo shell."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument
from hardware import KeyboardHardware

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="hardware",
        description="Shows or changes the default keyboard hardware used to generate layouts. ",
        arguments=(
            CommandArgument(
                "name",
                "[<name>]",
                "when specified, sets the default hardware to the given name in ./keebs/<name>.py. If not specified, shows the current hardware.",
            ),
        ),
        examples=("", "ansi", "ortho"),
        category="optimization",
        short_description="view or update the default keyboard hardware used to generate layouts",
    )


def complete(shell: "JaloShell", text: str, line: str, begidx: int, endidx: int) -> list[str]:
    # list every file in ./keebs/ that starts with the text and that ends with .py
    return [f.stem for f in Path("keebs").glob(f"{text}*.py")]


def exec(shell: "JaloShell", arg: str) -> None:
    if not arg:
        shell._info(f"Current hardware: {shell.hardware.name}")
        return

    try:
        new_hardware = KeyboardHardware.from_name(arg)
    except ValueError as e:
        shell._warn(f"error loading hardware: {e}")
        return

    shell._change_settings(hardware=new_hardware)
    shell._info(f"Updated hardware to: {new_hardware.name}")


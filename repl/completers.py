"""Auto-completion helpers and shared parsers for the Jalo shell."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from layout import KeyboardLayout

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    from repl.shell import JaloShell


def list_keyboard_names(shell: "JaloShell", prefix: str = "") -> list[str]:
    """Return all known keyboard identifiers, matching the optional prefix."""

    names: list[str] = []
    memory = shell.memory

    if not prefix or prefix[0].isdigit():
        names = [
            str(i)
            for i in range(1, 1 + len(memory.stack))
            if str(i).startswith(prefix)
        ]

        for list_num, layout_list in memory.lists.items():
            names.extend(
                [
                    f"{list_num}.{i}"
                    for i in range(1, 1 + len(layout_list.layouts))
                    if f"{list_num}.{i}".startswith(prefix)
                ]
            )

    names.extend([f.stem for f in Path("layouts").glob(f"{prefix}*.kb")])
    return names


def parse_keyboard_names(shell: "JaloShell", arg: str) -> list[KeyboardLayout] | None:
    """Parse one or more keyboard identifiers into concrete layouts."""

    names = shell._split_args(arg)
    if len(names) < 1:
        shell._warn("specify at least one keyboard layout")
        return None

    layouts: list[KeyboardLayout] = []
    memory = shell.memory

    for name in names:
        layout = None

        if "." in name:
            try:
                parts = name.split(".")
                if len(parts) != 2:
                    raise ValueError("Invalid format")
                list_num = int(parts[0])
                layout_idx = int(parts[1])

                if list_num not in memory.lists:
                    shell._warn(f"No list {list_num} found.")
                    return None
                if layout_idx < 1 or layout_idx > len(memory.lists[list_num].layouts):
                    count = len(memory.lists[list_num].layouts)
                    shell._warn(
                        f"No layout {name} found in list {list_num} (has {count} layouts)."
                    )
                    return None

                layout = memory.lists[list_num].layouts[layout_idx - 1]
                layouts.append(layout)
                continue
            except ValueError:
                pass
        else:
            try:
                stack_idx = int(name)
                if len(memory.stack) == 0:
                    shell._warn(
                        "No layouts in stack, so cannot retrieve '{name}'. Use 'generate', 'improve', or retrieve by name from ./layouts/."
                    )
                    return None
                if stack_idx < 1 or stack_idx > len(memory.stack):
                    shell._warn(
                        f"No layout {name} found in stack (has {len(memory.stack)} layouts)."
                    )
                    return None

                layout = memory.stack[stack_idx - 1]
                layouts.append(layout)
                continue
            except ValueError:
                pass

        try:
            hardware_name_hint = KeyboardLayout.hardware_hint(name)
        except FileNotFoundError as exc:
            shell._warn(f"could not find layout in: {exc.filename}")
            return None

        if hardware_name_hint:
            hardware = next(
                (hw for hw in shell.model_for_hardware if hw.name == hardware_name_hint),
                None,
            )
        else:
            hardware = shell.model.hardware

        try:
            layouts.append(KeyboardLayout.from_name(name, hardware))
        except Exception as exc:  # pragma: no cover - surface parse errors
            shell._warn(f"could not parse layout: {exc}")
            return None

    return layouts


__all__ = ["list_keyboard_names", "parse_keyboard_names"]

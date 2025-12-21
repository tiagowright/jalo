#!/usr/bin/env python
"""
Command-line REPL entry point for the Jalo keyboard tooling.

The shell currently exposes placeholder implementations for the available
commands and will be wired up to the real keyboard analysis helpers in future
iterations.
"""

import argparse
import cmd
import dataclasses
import shlex
import sys
import os
import re
from pathlib import Path
from typing import List, Optional
from inspect import cleandoc

from layout import KeyboardLayout
from model import KeyboardModel
from freqdist import FreqDist
from metrics import METRICS, use_oxeylyzer_mode
from objective import ObjectiveFunction
from hardware import KeyboardHardware, Hand
from optim import Optimizer
from repl.memory import LayoutMemoryManager
from repl.formatting import (
    format_table,
    format_analysis_table,
    format_command_help,
    format_help_summary,
    format_intro,
    format_layout_display,
    format_layout_memory,
    format_metrics_table,
    format_settings_info,
)


@dataclasses.dataclass(slots=True, frozen=True)
class CommandArgument:
    """Metadata for a command argument."""
    name: str
    syntax: str
    description: str


@dataclasses.dataclass(slots=True, frozen=True)
class Command:
    """Metadata for a shell command."""
    name: str
    description: str
    arguments: tuple[CommandArgument, ...]
    examples: tuple[str, ...]
    category: str  # e.g., "analysis", "editing", "configuration"
    short_description: str  # Short description from docstring, e.g., "analysis: analyze a keyboard layout on all metrics"



try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def _configure_readline() -> None:
    """Enable history and arrow-key navigation when readline is available."""
    try:
        import readline
    except ImportError:  # pragma: no cover - Windows fallback when readline missing
        return

    readline.set_history_length(5000)
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")

    history = Path.home() / ".jalo_history"
    try:
        readline.read_history_file(history)
    except FileNotFoundError:
        pass

    def _write_history() -> None:
        try:
            readline.write_history_file(history)
        except Exception:  # pragma: no cover - avoid crashing on filesystem issues
            pass

    import atexit

    atexit.register(_write_history)


@dataclasses.dataclass(slots=True, frozen=True)
class JaloSettings:
    hardware: str
    corpus: str
    oxeylyzer_mode: bool
    layouts_memory_size: int
    objective: str

    @classmethod
    def from_dict(cls, data: dict) -> "JaloSettings":
        hardware = data.get("hardware", "ortho")
        corpus = data.get("corpus", "en")
        oxeylyzer_mode = data.get("oxeylyzer_mode", False)
        objective = data.get("objective", "default")
        layouts_memory_size = data.get("layouts_memory_size", 100)
        return cls(hardware=str(hardware), corpus=str(corpus), oxeylyzer_mode=bool(oxeylyzer_mode), layouts_memory_size=int(layouts_memory_size), objective=str(objective))


class JaloShell(cmd.Cmd):
    """Interactive shell for exploring keyboard layouts."""

    prompt = "jalo> "
    # intro = "Jalo is just another keyboard layout optimizer – type 'help' to list commands."
    break_before_metrics = ['home','effort','lsb','same_hand','roll','redirect','left_hand','finger_0','sfb_finger_0']
    
    commands: dict[str, Command] = {}

    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path or Path(__file__).resolve().with_name("config.toml")

        self.memory = LayoutMemoryManager()
        self.model_for_hardware = {}
        self.current_command = ""  # Will be set by precmd before each command
        
        self._load_settings()

    def _load_settings(self):
        self.settings = _load_settings(self.config_path)
        self.freqdist = FreqDist.from_name(self.settings.corpus)
        self.metrics = METRICS
        use_oxeylyzer_mode(self.settings.oxeylyzer_mode)
        self.hardware = KeyboardHardware.from_name(self.settings.hardware)
        self.objective = ObjectiveFunction.from_name(self.settings.objective)

        self._load_objective(self.objective)
        self.pinned_chars = []

    def precmd(self, line: str) -> str:
        """Called before each command is executed. Save the command line."""
        self.current_command = line.strip()

        # remove comments (non-escaped # and everything after)
        line = re.sub(r'(?<!\\)#.*', '', line)

        return line

    def _load_settings_str(self):
        return format_settings_info(
            self.config_path,
            self.hardware.name,
            self.freqdist.corpus_name,
            str(self.objective),
        )

    def preloop(self) -> None:
        self._info(format_intro(self._load_settings_str()))


    def _load_objective(self, objective: ObjectiveFunction):
        self.objective = objective
        self.model = KeyboardModel(hardware=self.hardware, metrics=self.metrics, objective=self.objective, freqdist=self.freqdist)

        # with the new objective, need to invalidate all the cached models
        self.model_for_hardware = {
            self.hardware: self.model
        }

    def _get_model(self, layout: KeyboardLayout) -> KeyboardModel:
        if layout.hardware not in self.model_for_hardware:
            self.model_for_hardware[layout.hardware] = KeyboardModel(hardware=layout.hardware, metrics=self.metrics, objective=self.objective, freqdist=self.freqdist)
        return self.model_for_hardware[layout.hardware]

    

    # ----- core commands -------------------------------------------------
    commands["analyze"] = Command(
        name="analyze",
        description="Analyze the given keyboard layout, prints the layout, hardware, all metrics, and score.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to analyze, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0", "qwerty"),
        category="analysis",
        short_description="analyze a keyboard layout on all metrics",
    )

    def complete_analyze(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_analyze(self, arg: str) -> None:


        layouts = self._parse_keyboard_names(arg)
        if layouts is None or len(layouts)>1:
            self._warn("usage: analyze <keyboard>")
            return

        self.do_show(arg)
        self._info(self._tabulate_analysis(layouts))


    commands["show"] = Command(
        name="show",
        description="Shows the given keyboard layout, prints the layout and hardware.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to show, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0", "qwerty", "1.1"),
        category="analysis",
        short_description="show a keyboard layout",
    )

    def complete_show(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_show(self, arg: str) -> None:

        layouts = self._parse_keyboard_names(arg)
        if layouts is None or len(layouts) != 1:
            self._warn("usage: show <keyboard>")
            return

        self._info('')
        self._info(format_layout_display(layouts[0]))
        self._info('')


    commands["contributions"] = Command(
        name="contributions",
        description="Shows what is driving the score of one or more keyboard layouts, by tabulating the contributions that every metric makes to the total score.",
        arguments=(
            CommandArgument("keyboard", "<keyboard> [<keyboard>...]", "the keyboard layouts to analyze, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0 1", "0 sturdy qwerty graphite hdpm"),
        category="analysis",
        short_description="to understand the score, tabulates the contributions of each metric",
    )

    def complete_contributions(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_contributions(self, arg: str) -> None:
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: contributions <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts, show_contributions=True))



    commands["compare"] = Command(
        name="compare",
        description="Compares keyboard layouts side by side on every metric.",
        arguments=(
            CommandArgument("keyboard", "<keyboard> [<keyboard>...]", "the keyboard layouts to compare, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0 1", "0 sturdy qwerty graphite hdpm"),
        category="analysis",
        short_description="compares keyboard layouts side by side on every metric",
    )

    def complete_compare(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_compare(self, arg: str) -> None:
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts))
        
        

    commands["generate"] = Command(
        name="generate",
        description="Generates new layouts from scratch, starting from a number of random seeds and improving them to find the best (lowest) score, "
            "based on the current objective (see `help objective`). "
            "If a keyboard layout is provided, it will be used to determine the hardware and the location of pinned characters (see `help pin`). " 
            "If no keyboard layout is provided, default hardware is used and pinned characters are ignored with a warning.",
        arguments=(
            CommandArgument("seeds", "[seeds=100]", "the number of random seeds to generate, default is 100, which is enough for most cases."),
            CommandArgument("keyboard", "[<keyboard>]", "the keyboard layout to improve, can be a layout name or the index of a layout in memory."),
        ),
        examples=("", "1000", "100 hdpm"),
        category="optimization",
        short_description="generates a wide variety of new layouts from scratch",
    )

    def complete_generate(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        arg_num = self._arg_num_at_index(line, begidx, endidx)
        if arg_num is None or arg_num <= 1:
            seed_suggestions = ['10', '20', '100', '1000']
            return [suggestion for suggestion in seed_suggestions if suggestion.startswith(text)]

        return self._list_keyboard_names(text)

    def do_generate(self, arg: str) -> None:
        args = self._split_args(arg)

        seeds = 100
        layout = None

        if len(args) > 0:
            try:
                seeds = int(args[0])
            except ValueError:
                self._warn("seeds must be an integer: generate [seeds=100] [keyboard]")
                return
        
        if len(args) > 1:
            layouts = self._parse_keyboard_names(args[1])
            if layouts is None or len(layouts) != 1:
                self._warn("usage: generate <seeds> [keyboard]")
                return
            layout = layouts[0]

        if layout is not None:
            model = self._get_model(layout)
            pinned_positions = model.pinned_positions_from_layout(layout, self.pinned_chars)
        else:
            model = self.model
            pinned_positions = ()
            if self.pinned_chars:
                self._warn(f"no keyboard layout provided, cannot pin characters to a position, pins will be ignored ({', '.join(self.pinned_chars)})")

        self._info(f"generating {seeds} seeds.")

        optimizer = Optimizer(model, population_size=100, solver='genetic')

        if layout is not None:
            optimizer.generate(
                seeds=seeds, 
                char_at_pos=tuple(model.char_at_positions_from_layout(layout)), 
                pinned_positions=pinned_positions
            )
        else:            
            N = len(model.hardware.positions)
            char_seq = self.freqdist.char_seq[:N]
            optimizer.generate(char_seq=char_seq, seeds=seeds, pinned_positions=pinned_positions)
        
        list_num = self._layout_memory_from_optimizer(optimizer, push_to_stack=False)

        self._info(f'')
        # Show the newly created list
        self._info(self._layout_memory_to_str(list_num=list_num))



    commands["improve"] = Command(
        name="improve",
        description="Tries to improve the score of a given layout by swapping positions and columns. "
        "Produces layouts that are not very different from the original layout, but that score better if possible. "
        "While `generate` produces a wide variety of very different layouts that are a minimum distance from each other, "
        "`improve` does the opposite, producing layouts that are within a maximum distance. So these are complementary steps, "
        "with `generate` providing large coarse grained cuts, and `improve` refining it further, and finally `polish` helping with the final "
        "few swaps.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to improve, can be a layout name or the index of a layout in memory."),
            CommandArgument("seeds", "[seeds=100]", "the number of random seeds to generate, default is 100, which is enough for most cases."),
        ),
        examples=("0", "qwerty", "hdpm 200"),
        category="optimization",
        short_description="try to improve the score of a given layout (neighboring layouts)",
    )

    def complete_improve(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        arg_num = self._arg_num_at_index(line, begidx, endidx)
        if arg_num is None or arg_num <= 1:
            return self._list_keyboard_names(text)

        seed_suggestions = ['10', '20', '100', '200', '1000']
        return [suggestion for suggestion in seed_suggestions if suggestion.startswith(text)]

    def do_improve(self, arg: str) -> None:
        args = self._split_args(arg)

        if not args:
            self._warn("usage: improve <keyboard> [seeds=100]")
            return

        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("usage: improve <keyboard>")
            return

        if len(args) > 1:
            try:
                seeds = int(args[1])
            except ValueError:
                self._warn("iterations must be an integer: improve <keyboard> [seeds=100]")
                return
        else:
            seeds = 100


        self._info(f"improving {layouts[0].name} with {seeds} seeds.")

        layout = layouts[0]
        model = self._get_model(layout)
        pinned_positions = model.pinned_positions_from_layout(layout, self.pinned_chars)

        optimizer = Optimizer(model, population_size=100, solver = 'annealing')
        optimizer.improve(
            char_at_pos=tuple(model.char_at_positions_from_layout(layout)), 
            seeds=seeds,
            hamming_distance_threshold=10,
            pinned_positions=pinned_positions
        )
        
        list_num = self._layout_memory_from_optimizer(optimizer, original_layout=layout, push_to_stack=False)

        self._info(f'')
        # Show the newly created list
        self._info(self._layout_memory_to_str(list_num=list_num))



    commands["polish"] = Command(
        name="polish",
        description="Identifies small number of swaps that can improve the score of a given layout.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to polish, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0", "qwerty"),
        category="optimization",
        short_description="identifies small number of swaps that can improve the score of a given layout",
    )

    def complete_polish(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_polish(self, arg: str) -> None:
        args = self._split_args(arg)
        layouts = self._parse_keyboard_names(args[0])

        if layouts is None or len(layouts) != 1:
            self._warn("usage: polish <keyboard>")
            return
        
        layout = layouts[0]
        model = self._get_model(layout)
        char_at_pos = tuple(model.char_at_positions_from_layout(layout))
        pinned_positions = model.pinned_positions_from_layout(layout, self.pinned_chars)

        optimizer = Optimizer(model, population_size=100, solver = 'annealing')
        swaps_scores = optimizer.polish(
            char_at_pos=char_at_pos,
            iterations=100,
            max_depth=3,
            pinned_positions=pinned_positions
        )
        
        # list_num = self._layout_memory_from_optimizer(optimizer, original_layout=layout, push_to_stack=False)

        by_len = {}
        for swaps, score in swaps_scores.items():
            len_swaps = len(swaps)
            if len_swaps == 0:
                continue
            if len_swaps not in by_len:
                by_len[len_swaps] = []
            by_len[len_swaps].append((swaps, score))


        parser = shlex.shlex()

        for n_swaps in sorted(by_len.keys()):
            self._info(f'{n_swaps} swaps:')
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
                    chars.append(f'{chari}{charj}')
                    swapped_char_at_pos[i], swapped_char_at_pos[j] = swapped_char_at_pos[j], swapped_char_at_pos[i]
                
                chars_str = ' '.join(chars)
                rows.append([f'swap {layout.name} {chars_str}', f'# score: {100*score:.2f}'])
            self._info(format_table(rows))
            self._info('')

        self._info('')


    commands["swap"] = Command(
        name="swap",
        description="Swaps position pairs on the keyboard, for those hand crafted, fine tuning adjustments to a layout.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to swap positions on, can be a layout name or the index of a layout in memory."),
            CommandArgument("pair", "<pair> [<pair> ...]", "the character pairs to swap positions"),
        ),
        examples=("0 dh", "qwerty aq wz"),
        category="editing",
        short_description="swaps two or more positions on the keyboard",
    )

    def complete_swap(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        arg_num = self._arg_num_at_index(line, begidx, endidx)
        if arg_num is None or arg_num <= 1:
            return self._list_keyboard_names(text)

        return [char for char in self.model.freqdist.char_seq if char != self.model.freqdist.out_of_distribution]


    def do_swap(self, arg: str) -> None:
        args = self._split_args(arg)

        if len(args) < 2:
            self._warn("usage: swap <keyboard> <pair> [<pair> ...]")
            return
        
        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("usage: swap <keyboard> <pair> [<pair> ...]")
            return
        
        layout = layouts[0]
        
        pairs = args[1:]
        for pair in pairs:
            if len(pair) != 2:
                self._warn("each pair must be two characters, could not understand pair: {pair}")
                return
            
            char1, char2 = pair
            if char1 not in layout.char_to_key:
                self._warn(f"character {char1} not found in layout")
                return
            if char2 not in layout.char_to_key:
                self._warn(f"character {char2} not found in layout")
                return

        swapped_layout = layout.swap(char_pairs=[(char1, char2) for char1, char2 in pairs])

        self._push_layout_to_stack(swapped_layout, layout)

        self._info(f'')
        self._info(self._layout_memory_to_str(list_num=0, top_n=1))


    commands["mirror"] = Command(
        name="mirror",
        description="Mirrors a keyboard layout horizontally, so that the left and right hands are swapped.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to mirror, can be a layout name or the index of a layout in memory."),
        ),
        examples=("0", "hdpm"),
        category="editing",
        short_description="mirrors a keyboard layout horizontally",
    )

    def complete_mirror(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_mirror(self, arg: str) -> None:
        args = self._split_args(arg)

        if len(args) != 1:
            self._warn("usage: mirror <keyboard>")
            return

        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("usage: mirror <keyboard>")
            return
        
        layout = layouts[0]
        mirrored_layout = layout.mirror()

        self._push_layout_to_stack(mirrored_layout, layout)

        self._info(f'')
        self._info(self._layout_memory_to_str(list_num=0, top_n=1))


    commands["invert"] = Command(
        name="invert",
        description="Inverts top and bottom rows of a keyboard layout (mirrors vertically).",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to invert, can be a layout name or the index of a layout in memory."),
            CommandArgument("hand", "[hand=both]", "`left` or `right` to invert the left or right hand, default is `both`."),
        ),
        examples=("0", "hdpm right"),
        category="editing",
        short_description="inverts top and bottom rows of a keyboard layout (mirrors vertically)",
    )
    def complete_invert(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        arg_num = self._arg_num_at_index(line, begidx, endidx)
        if arg_num is None or arg_num <= 1:
            return self._list_keyboard_names(text)

        return [word for word in ('left', 'right', 'both') if word.startswith(text)]

    def do_invert(self, arg: str) -> None:
        args = self._split_args(arg)

        if len(args) < 1:
            self._warn("specify a keyboard. Usage: invert <keyboard> [hand=both]")
            return

        if len(args) > 2:
            self._warn("too many arguments. Usage: invert <keyboard> [hand=both]")
            return

        if len(args) >= 2:
            hand_str = args[1].lower()
            if hand_str not in ('left', 'right', 'both'):
                self._warn("invalid hand: {hand_str}, specify `left` or `right` or `both`. Usage: invert <keyboard> [hand=both]")
                return
        else:
            hand_str = 'both'

        hand = None if hand_str == 'both' else Hand.LEFT if hand_str == 'left' else Hand.RIGHT

        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("invalid keyboard: {args[0]}. Usage: invert <keyboard> [hand=both]")
            return
        
        layout = layouts[0]
        inverted_layout = layout.invert(hand=hand)

        self._push_layout_to_stack(inverted_layout, layout)

        self._info(f'')
        self._info(self._layout_memory_to_str(list_num=0, top_n=1))



    commands["pin"] = Command(
        name="pin",
        description="Pins characters to their current positions, so that they cannot be swapped or moved to a different position during editing (`generate`, `improve`, etc.). "
        "To see current pins, pass no arguments (`pin`). To clear all pins, use `pin nothing`.",
        arguments=(
            CommandArgument("`nothing`", "[nothing]", "to remove all existing pins, specify `pin nothing`."),
            CommandArgument("chars", "[<chars> ...]", "the characters to pin, can be a single, multiple characters, or a space-separated list of characters."),
        ),
        examples=("", "a b c", "asdf", "aeiou", "nothing"),
        category="optimization",
        short_description="pins characters to their current position",
    )

    def complete_pin(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        words = [word for word in ('nothing', 'clear') if word.startswith(text)]

        if words and endidx-begidx >= 2:
            return words

        return words + [char for char in self.model.freqdist.char_seq if char != self.model.freqdist.out_of_distribution] 


    def do_pin(self, arg: str) -> None:
        args = self._split_args(arg)
        
        if not args:
            pass

        elif args[0].lower() in ('nothing', 'clear'):
            self.pinned_chars = []

        else:
            all_chars = [char for chars in args for char in chars]

            invalid_chars = [char for char in all_chars if char not in self.model.freqdist.char_seq]
            if invalid_chars:
                str_invalid_chars = ' '.join(invalid_chars)
                self._warn(f"Warning: {len(invalid_chars)} character(s) are not in the corpus: {str_invalid_chars}")

            self.pinned_chars.extend(all_chars)

        # display pinned characters
        if not self.pinned_chars:
            self._info("nothing pinned.")
        else:
            str_pinned_chars = ' '.join(self.pinned_chars)
            self._info(f"pinned: {str_pinned_chars}")



    commands["list"] = Command(
        name="list",
        description="Lists layouts in memory. Called with no arguments, shows available lists and top 3 layouts from the stack. Called with a list number, shows layouts from that list. Called with two arguments, the second is the number of layouts to show.",
        arguments=(
            CommandArgument("list_num", "[<list_num>]", "the list number to show (0 for stack, >0 for numbered lists). If omitted, shows all lists and top 3 from stack."),
            CommandArgument("count", "[<count>=10]", "the number of layouts to show, default is 10 for numbered lists, 3 for stack when no arguments."),
        ),
        examples=("", "0", "1", "2 5"),
        category="editing",
        short_description="lists layouts in memory",
    )

    def do_list(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        args = self._split_args(arg)
        
        if len(args) == 0:
            # No arguments: list available lists, then top 3 from stack
            self._info("Available lists:")
            if len(self.memory.stack) > 0:
                self._info(f"  stack: {len(self.memory.stack)} layouts")
            else:
                self._info("  stack: (empty)")
            
            list_numbers = self.memory.get_all_list_numbers()
            if list_numbers:
                for list_num in list_numbers:
                    if list_num in self.memory.lists:
                        count = len(self.memory.lists[list_num].layouts)
                        command = self.memory.lists[list_num].command
                        self._info(f"  {list_num}: {count} layouts  > {command}")
            else:
                self._info("  (no numbered lists)")
            
            self._info("")
            self._info("Top 3 layouts from stack:")
            self._info(self._layout_memory_to_str(list_num=0, top_n=3))
            return
        
        # Parse list number
        try:
            list_num = int(args[0])
        except ValueError:
            self._warn(f"list number must be an integer: list [<list_num>] [<count>]")
            return
        
        # Parse count (default: 10 for numbered lists, 3 for stack)
        if len(args) > 1:
            try:
                count = int(args[1])
            except ValueError:
                self._warn(f"count must be an integer: list [<list_num>] [<count>]")
                return
        else:
            count = 10 if list_num > 0 else 3
        
        # Show the requested list
        self._info(self._layout_memory_to_str(list_num=list_num, top_n=count))



    commands["reload"] = Command(
        name="reload",
        description="Reload the settings from the config.toml file. Keeps generated layouts results in memory, but updates corpus, hardware, and objective function.",
        arguments=(),
        examples=("",),
        category="configuration",
        short_description="reload the settings from the config.toml file",
    )

    def do_reload(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        # should reload the settings from the config.toml file
        self._load_settings()

        

    commands["objective"] = Command(
        name="objective",
        description="\n".join([
            "Shows the current objective function when called without an argument, or sets the objective function" 
            "to a new formula when that is given. The objective function is considered an effort, meaning lower scores are better.",
            "",
            "The formula is any linear combination of metrics:"
            "  [+|-][weight_1]<metric_name_1> [+|- [weight_2]<metric_name_2> ...]",
            "",
            "where:",
            "  <weight_i> is a float.",
            "  <metric_name_i> is the name of a valid metric (type `metrics`)",
            "",
            "Note that there is no `*` between the weight and the metric name."
        ]),
        arguments=(
            CommandArgument("formula", "[<formula>]", "when specified, sets the objective function to the given formula."),
        ),
        examples=("", "100sfb + 6effort + 60pinky_ring + 60scissors_ortho + 60sfs - 5alt"),
        category="configuration",
        short_description="view or update the objective function used to score layouts",
    )

    def complete_objective(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        sign_pattern = re.compile(r'[+-]')
        float_pattern = re.compile(r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

        # if text begins with sign_pattern and/or float_pattern, remove them first
        i = 0
        match = sign_pattern.match(text, i)
        if match:
            i = match.end()

        match = float_pattern.match(text, i)
        if match:
            i = match.end()

        # find all metric names that start with text
        return [text[:i] + metric.name for metric in self.metrics if metric.name.startswith(text[i:])]

    def do_objective(self, arg: str) -> None:
        
        if not arg:
            self._info(f"Current function:\nobjective {self.objective}\n")
            return
            
        try:
            new_objective = ObjectiveFunction.from_formula(arg)
        except ValueError as e:
            self._warn(f"error parsing objective function: {e}")
            return

        self._load_objective(new_objective)
        self._info(f"Updated function to:\nobjective {self.objective}\n")



    commands["metrics"] = Command(
        name="metrics",
        description="Shows all current metrics and their descriptions.",
        arguments=(),
        examples=("",),
        category="configuration",
        short_description="shows metric names and descriptions",
    )

    def do_metrics(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        self._info(format_metrics_table(self.metrics, set(self.break_before_metrics)))


    commands["save"] = Command(
        name="save",
        description="Saves a layout to the ./layouts directory. Saved layouts can then be used in other commands by name.",
        arguments=(
            CommandArgument("keyboard", "<keyboard>", "the keyboard layout to save, can be a layout name or the index of a layout in memory."),
            CommandArgument("name", "[<name>]", "the name of the new layout, defaults to the home row characters."),
        ),
        examples=("0", "1 mylayout"),
        category="editing",
        short_description="save new layouts from memory to a new file",
    )

    def complete_save(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_save(self, arg: str) -> None:
        args = self._split_args(arg)
        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("usage: save <keyboard> [<name>]")
            return
        
        layout = layouts[0]

        if len(args) > 1:
            name_candidate = args[1]
        else:
            # otherwise, use the home key characters
            name_candidate = ''.join(key.char for key in layout.keys if key.position.is_home)

        filename = f"{name_candidate}.kb"
        filepath = os.path.join('layouts', filename)

        if os.path.exists(filepath):
            self._warn(f"layout file already exists: {filepath}, not overwriting. Specify a different name: save <keyboard> <name>")
            return

        with open(filepath, 'w') as f:
            f.write(str(layout))

        self._info(f"saved layout to {filepath}")



    # ----- shell controls ------------------------------------------------
    def do_help(self, arg: str) -> None:  # type: ignore[override]
        """help with any command, try `help analyze`"""
        if arg:
            # Look up command in the commands dict
            if arg in self.commands:
                cmd = self.commands[arg]
                argument_name_description = [
                    (arg.name, arg.syntax, arg.description)
                    for arg in cmd.arguments
                ]
                self._info(format_command_help(
                    command=cmd.name,
                    description=cmd.description,
                    argument_name_description=argument_name_description,
                    examples=list(cmd.examples),
                ))
            else:
                # Fallback to default behavior for commands not in dict
                super().do_help(arg)
            return
        
        # Build summary from commands dict and docstrings
        names = self.get_names()

        headers = ['analysis', 'optimization', 'editing', 'configuration', 'commands']
        commands_list = {header: list() for header in headers}

        for name in names:
            if name[:3] != 'do_':
                continue
            
            cmd_name = name[3:]
            if cmd_name in self.commands:
                cmd = self.commands[cmd_name]
                if cmd.category not in commands_list:
                    commands_list[cmd.category] = []
                    headers.append(cmd.category)
                commands_list[cmd.category].append((cmd_name, cmd.short_description))
            else:
                # Fallback to docstring for commands not in dict
                try:
                    doc = getattr(self, name).__doc__
                    doc = cleandoc(doc)
                except AttributeError:
                    doc = ''
                commands_list['commands'].append((cmd_name, doc))


        self._info(format_help_summary(headers, commands_list))


    commands["quit"] = Command(
        name="quit",
        description="Quits Jalo.",
        arguments=(),
        examples=("",),
        category="commands",
        short_description="quits Jalo",
    )

    commands["exit"] = Command(
        name="exit",
        description="Quits Jalo.",
        arguments=(),
        examples=("",),
        category="commands",
        short_description="quits Jalo",
    )

    def do_quit(self, arg: str) -> bool: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        self._info("Exiting Jalo REPL. Bye bye.")
        self._info("")
        return True

    def do_exit(self, arg: str) -> bool:
        return self.do_quit(arg)


    # ----- helpers -------------------------------------------------------
    def _info(self, message: str) -> None:
        print(message)

    @staticmethod
    def _warn(message: str) -> None:
        print(message, file=sys.stderr)
    

    @staticmethod
    def _split_args(arg: str) -> List[str]:
        try:
            return shlex.split(arg)
        except ValueError as exc:
            print(f"error parsing arguments: {exc}", file=sys.stderr)
            return []

    def _arg_num_at_index(self, line, begidx: int, endidx: int) -> int | None:
        '''return the count of arguments in the line before begidx, that is, the argument number at begidx'''
        try:
            args = shlex.split(line[:begidx])
        except ValueError as exc:
            return None
        return len(args)

    def _list_keyboard_names(self, prefix: str = '') -> list[str]:
        names = []
        if not prefix or prefix[0].isdigit():
            names = [str(i) for i in range(1, 1+len(self.memory.stack)) if str(i).startswith(prefix)]
            
            for list_num in self.memory.lists.keys():
                names.extend([f'{list_num}.{i}' for i in range(1, 1+len(self.memory.lists[list_num].layouts)) if f'{list_num}.{i}'.startswith(prefix)])
        
        names.extend([
            f.stem
            for f in Path('layouts').glob(f'{prefix}*.kb')
        ])

        return names


    def _parse_keyboard_names(self, arg: str) -> list[KeyboardLayout] | None:
        names = self._split_args(arg)
        if len(names) < 1:
            self._warn("specify at least one keyboard layout")
            return None

        layouts = []

        for name in names:
            layout = None
            
            # Try to parse as memory reference: "1.2" (list 1, layout 2) or "3" (stack layout 3)
            if '.' in name:
                # Format: list_num.layout_idx (e.g., "1.2")
                try:
                    parts = name.split('.')
                    if len(parts) != 2:
                        raise ValueError("Invalid format")
                    list_num = int(parts[0])
                    layout_idx = int(parts[1])
                    
                    if list_num not in self.memory.lists:
                        self._warn(f"No list {list_num} found.")
                        return None
                    if layout_idx < 1 or layout_idx > len(self.memory.lists[list_num].layouts):
                        count = len(self.memory.lists[list_num].layouts)
                        self._warn(f"No layout {name} found in list {list_num} (has {count} layouts).")
                        return None
                    
                    layout = self.memory.lists[list_num].layouts[layout_idx - 1]
                    layouts.append(layout)
                    continue
                except ValueError:
                    pass  # Not a memory reference, try other formats
            else:
                # Try as stack reference: single integer (e.g., "3")
                try:
                    stack_idx = int(name)
                    if len(self.memory.stack) == 0:
                        self._warn(f"No layouts in stack, so cannot retrieve '{name}'. Use 'generate', 'improve', or retrieve by name from ./layouts/.")
                        return None
                    if stack_idx < 1 or stack_idx > len(self.memory.stack):
                        self._warn(f"No layout {name} found in stack (has {len(self.memory.stack)} layouts).")
                        return None
                    
                    layout = self.memory.stack[stack_idx - 1]
                    layouts.append(layout)
                    continue
                except ValueError:
                    pass  # Not an int, try loading by name

            # Try loading by name from file
            try:
                # check if layout specifies the hardware
                hardware_name_hint = KeyboardLayout.hardware_hint(name)
            
            except FileNotFoundError as e:
                self._warn(f"could not find layout in: {e.filename}")
                return None

            if hardware_name_hint:
                for hardware in self.model_for_hardware:
                    if hardware.name == hardware_name_hint:
                        break
                else:
                    hardware = None
            else:
                # no hint found, try the default hardware
                hardware = self.model.hardware

            try:
                layouts.append(KeyboardLayout.from_name(name, hardware))

            except Exception as e:
                self._warn(f"could not parse layout: {e}")
                return None

        return layouts

    def _layout_memory_to_str(self, list_num: int | None = None, top_n: int = 10) -> str:
        """Format layouts from memory as a string."""
        def score_fn(layout: KeyboardLayout) -> float:
            return self._get_model(layout).score_layout(layout)
        
        return format_layout_memory(self.memory, list_num, top_n, score_fn)

    def _layout_memory_from_optimizer(self, optimizer: Optimizer, original_layout: KeyboardLayout | None = None, push_to_stack: bool = False) -> int | None:
        """Add layouts from optimizer to memory.
        
        Args:
            optimizer: The optimizer containing the layouts
            original_layout: Original layout if applicable
            push_to_stack: If True, push to stack. If False, create a new numbered list.
            
        Returns:
            The list number if a new list was created, None if pushed to stack
        """
        layouts = []
        for i, new_char_at_pos in enumerate(optimizer.population.sorted()[:10]):
            new_layout = optimizer.model.layout_from_char_at_positions(new_char_at_pos, original_layout=original_layout, name = f'')
            layouts.append(new_layout)
        
        if push_to_stack:
            # Push each layout to stack individually
            for layout in layouts:
                self.memory.push_to_stack(layout, self.current_command, original_layout)
            return None
        else:
            # Create a new numbered list
            return self.memory.add_list(layouts, self.current_command, original_layout)
    
    def _push_layout_to_stack(self, layout: KeyboardLayout, original_layout: KeyboardLayout | None = None) -> None:
        """Push a single layout to the stack.
        
        Args:
            layout: The layout to push to the stack
            original_layout: The original layout if one was given as argument
        """
        self.memory.push_to_stack(layout, self.current_command, original_layout)


    def _tabulate_analysis(self, layouts: List[KeyboardLayout], show_contributions: bool = False) -> str:
        model_for_layout = {layout: self._get_model(layout) for layout in layouts}
        show_top_ngram = len(layouts) == 1

        try:
            analysis = {layout: model_for_layout[layout].analyze_layout(layout) for layout in layouts}
            scores = {layout: model_for_layout[layout].score_layout(layout) for layout in layouts}
            contributions = {layout: model_for_layout[layout].score_contributions(layout) for layout in layouts} if show_contributions else None
            top_ngram = {layout: model_for_layout[layout].top_ngram_per_metric(layout) for layout in layouts} if show_top_ngram else None

        except Exception as e:
            self._warn(f"Error: could not analyze layout: {e}")
            return ''

        return format_analysis_table(
            layouts=layouts,
            metrics=self.metrics,
            analysis=analysis,
            scores=scores,
            contributions=contributions,
            top_ngram=top_ngram,
            objective_weights=self.model.objective.metrics if show_contributions else None,
            break_before_metrics=set(self.break_before_metrics),
            show_contributions=show_contributions,
        )





DEFAULT_CONFIG = '''
hardware = "ortho"
corpus = "en"
'''

def _load_settings(config_path: Path) -> JaloSettings:
    if not config_path.exists():
        print(f"warning: config file not found at {config_path}, creating default config", file=sys.stderr)
        try:
            config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO errors should not kill REPL
            print(f"Fatal error: failed to create config {config_path}: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:  # pragma: no cover - IO errors should not kill REPL
        print(f"Fatal error: failed to read config {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(data, dict):
        print(f"Fatal error: malformed config in {config_path}", file=sys.stderr)
        sys.exit(1)

    return JaloSettings.from_dict(data)




def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Jalo keyboard layout analyzer and optimizer."
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable readline history persistence.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a keyboard layout.")
    analyze_parser.add_argument("keyboard", help="The name of the keyboard layout to analyze.")

    # Contributions command
    contributions_parser = subparsers.add_parser("contributions", help="Show contributions of a layout.")
    contributions_parser.add_argument("keyboards", nargs="+", help="The names of the keyboard layouts.")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a new keyboard layout.")
    generate_parser.add_argument("iterations", type=int, nargs='?', default=100, help="Number of iterations.")

    # Improve command
    improve_parser = subparsers.add_parser("improve", help="Improve an existing keyboard layout.")
    improve_parser.add_argument("keyboard", help="The name of the keyboard layout to improve.")
    improve_parser.add_argument("iterations", type=int, nargs='?', default=10, help="Number of iterations.")

    # Metrics command
    subparsers.add_parser("metrics", help="Show metrics of a layout.")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two or more layouts.")
    compare_parser.add_argument("keyboards", nargs="+", help="The names of the keyboard layouts to compare.")

    args = parser.parse_args(argv)

    if not args.no_history:
        _configure_readline()

    config_path = Path(__file__).resolve().with_name("config.toml")
    try:
        shell = JaloShell(config_path=config_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.command:
        if args.command == "analyze":
            shell.do_analyze(args.keyboard)
        elif args.command == "contributions":
            shell.do_contributions(" ".join(args.keyboards))
        elif args.command == "generate":
            shell.do_generate(f"{args.iterations}")
        elif args.command == "improve":
            shell.do_improve(f"{args.keyboard} {args.iterations}")
        elif args.command == "metrics":
            shell.do_metrics("")
        elif args.command == "compare":
            shell.do_compare(" ".join(args.keyboards))
    else:
        # shell._info(
        #     f"Loaded config: hardware='{shell.settings.hardware}', "
        #     f"corpus='{shell.settings.corpus}'."
        # )
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            shell._info("")  # ensure a newline after Ctrl+C
            shell._info("Interrupted. Bye bye.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

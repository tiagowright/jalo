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
import textwrap
import re
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate
from inspect import cleandoc

from layout import KeyboardLayout
from model import KeyboardModel
from freqdist import FreqDist
from metrics import METRICS, Metric, use_oxeylyzer_mode
from objective import ObjectiveFunction
from hardware import KeyboardHardware
from optim import Optimizer, Population



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

    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path or Path(__file__).resolve().with_name("config.toml")

        # keep a sorted list of the top generated layouts by score
        self.layouts_memory = []
        self.model_for_hardware = {}
        
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

    def _load_settings_str(self):
        return "\n".join([
            f"loaded settings from {self.config_path}:",
            f"hardware = '{self.hardware.name}'",
            f"corpus = '{self.freqdist.corpus_name}'",
            f"objective = '{self.objective}'",
            ""
        ])

    def preloop(self) -> None:
        self._info("\n".join([
            "",
            "",
            "--",
            "Jalo -- just another layout optimizer",
            "--",
            "",
            self._load_settings_str(),
            "",
            "Tab does autocomplete, up arrow brings up previous commands. Type `help` to list commands.",
            ""
        ]))


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
    def help_analyze(self) -> None:
        self._command_help(
            command="analyze",
            description="Analyze the given keyboard layout, prints the layout, hardware, all metrics, and score.",
            argument_name_description=[
                ("keyboard", "<keyboard>", "the keyboard layout to analyze, can be a layout name or the index of a layout in memory.")
            ],
            examples=["0", "qwerty"],
        )

    def complete_analyze(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_analyze(self, arg: str) -> None:
        """analysis: analyze a keyboard layout on all metrics"""


        layouts = self._parse_keyboard_names(arg)
        if layouts is None or len(layouts)>1:
            self._warn("usage: analyze <keyboard>")
            return

        # header = [layouts[0].name, layouts[0].hardware.name]
        # layout_str = str(layouts[0])
        # hardware_str = layouts[0].hardware.str(show_finger_numbers=True, show_stagger=True)

        # # annoyingly, tabulate removes leading spaces and in this case screws up the formating of layouts
        # # so adding a leading character
        # LEAD_SPACE = "| "
        # rows = zip(
        #     [LEAD_SPACE + l for l in layout_str.split('\n')], 
        #     [LEAD_SPACE + l for l in hardware_str.split('\n')]
        # )

        # self._info('')
        # self._info(tabulate(rows, headers=header, tablefmt="simple", disable_numparse=True))
        # self._info('')
        self.do_show(arg)
        self._info(self._tabulate_analysis(layouts))


    def help_show(self) -> None:
        self._command_help(
            command="show",
            description="Shows a keyboard layout's key assignments and hardware finger assignments.",
            argument_name_description=[("keyboard", "<keyboard>", "the keyboard layout to show, can be a layout name or the index of a layout in memory.")],
            examples=["0", "qwerty"],
        )
        
    def complete_show(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_show(self, arg: str) -> None:
        """analysis: shows a keyboard layout"""

        layouts = self._parse_keyboard_names(arg)
        if layouts is None or len(layouts) != 1:
            self._warn("usage: show <keyboard>")
            return

        header = [layouts[0].name, layouts[0].hardware.name]
        layout_str = str(layouts[0])
        hardware_str = layouts[0].hardware.str(show_finger_numbers=True, show_stagger=True)

        # annoyingly, tabulate removes leading spaces and in this case screws up the formating of layouts
        # so adding a leading character
        LEAD_SPACE = "| "
        rows = zip(
            [LEAD_SPACE + l for l in layout_str.split('\n')], 
            [LEAD_SPACE + l for l in hardware_str.split('\n')]
        )

        self._info('')
        self._info(tabulate(rows, headers=header, tablefmt="simple", disable_numparse=True))
        self._info('')


    def help_contributions(self) -> None:
        self._command_help(
            command="contributions",
            description="Shows what is driving the score of one or more keyboard layouts, by tabulating the contributions that every metric makes to the total score.",
            argument_name_description=[
                ("keyboard", "<keyboard> [<keyboard>...]", "the keyboard layouts to analyze, can be a layout name or the index of a layout in memory.")                
            ],
            examples=["0 1", "0 sturdy qwerty graphite hdpm"],
        )

    def complete_contributions(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_contributions(self, arg: str) -> None:
        """analysis: to understand the score, tabulates the contributions of each metric"""
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: contributions <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts, show_contributions=True))



    def help_compare(self) -> None:
        self._command_help(
            command="compare",
            description="Compares keyboard layouts side by side on every metric.",
            argument_name_description=[
                ("keyboard", "<keyboard> [<keyboard>...]", "the keyboard layouts to compare, can be a layout name or the index of a layout in memory.")
            ],
            examples=["0 1", "0 sturdy qwerty graphite hdpm"],
        )
        
    def complete_compare(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)
 
    def do_compare(self, arg: str) -> None:
        """analysis: compares keyboard layouts side by side on every metric"""
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts))
        
        

    def help_generate(self) -> None:
        self._command_help(
            command="generate",
            description="Generates new layouts from scratch, starting from a number of random seeds and improving them to find the best (lowest) score, based on the current objective (see `help objective`).",
            argument_name_description=[("seeds", "[seeds=100]", "the number of random seeds to generate, default is 100.")],
            examples=["", "1000"],
        )

    def do_generate(self, arg: str) -> None:
        """editing: generates a wide variety of new layouts from scratch"""
        args = self._split_args(arg)

        seeds = 100

        if len(args) > 0:
            try:
                seeds = int(args[0])
            except ValueError:
                self._warn("seeds must be an integer: generate [seeds=100]")
                return
        
        self._info(f"generating {seeds} seeds.")

        N = len(self.model.hardware.positions)
        char_seq = self.freqdist.char_seq[:N]

        optimizer = Optimizer(self.model, population_size=100)
        optimizer.generate(char_seq=char_seq, seeds=seeds)
        
        self._layout_memory_from_optimizer(optimizer)

        self._info(f'')
        self._info(self._layout_memory_to_str())



    def help_improve(self) -> None:
        self._command_help(
            command="improve",
            description="Tries to improve the score of a given layout by swapping positions and columns. "
            "Produces layouts that are not very different from the original layout, but that score better if possible. "
            "While `generate` produces a wide variety of very different layouts that are a minimum distance from each other, "
            "`improve` does the opposite, producing layouts that are within a maximum distance. So these are complementary steps, "
            "with `generate` providing large coarse grained cuts, and `improve` refining it further, and finally `polish` helping with the final "
            "few swaps.",
            argument_name_description=[("keyboard", "<keyboard>", "the keyboard layout to improve, can be a layout name or the index of a layout in memory.")],
            examples=["0", "qwerty"],
        )
        
    def complete_improve(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_improve(self, arg: str) -> None:
        """editing: try to improve the score of a given layout (neighboring layouts)"""
        args = self._split_args(arg)

        layouts = self._parse_keyboard_names(args[0])
        if layouts is None or len(layouts) != 1:
            self._warn("usage: improve <keyboard>")
            return

        if len(args) > 1:
            try:
                iterations = int(args[1])
            except ValueError:
                self._warn("iterations must be an integer: improve <keyboard> [iterations=10]")
                return
        else:
            iterations = 10
        
        layout = layouts[0]
        
        model = self._get_model(layout)
        original_char_at_pos = model.char_at_positions_from_layout(layout)
        pinned_positions = model.pinned_positions_from_layout(layout, self.pinned_chars)
        char_at_pos = original_char_at_pos.copy()
        original_score = model.score_chars_at_positions(char_at_pos)

        self._info(f"improving {layout.name} {original_score*100:.3f}...")
        
        optimizer = Optimizer(model, population_size=10)
        optimizer.optimize(char_at_pos, iterations=iterations, pinned_positions=pinned_positions)

        if len(optimizer.population) == 0:
            self._warn("no improvement found, no layouts added to memory")
            return

        self._layout_memory_from_optimizer(optimizer, original_layout=layout)

        self._info(f'')
        self._info(self._layout_memory_to_str(original_score=original_score))



    def help_polish(self) -> None:
        self._command_help(
            command="polish",
            description="Identifies small number of swaps that can improve the score of a given layout.",
            argument_name_description=[("keyboard", "<keyboard>", "the keyboard layout to polish, can be a layout name or the index of a layout in memory.")],
            examples=["0", "qwerty"],
        )

    def complete_polish(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_polish(self, arg: str) -> None:
        """editing: identifies small number of swaps that can improve the score of a given layout"""
        args = self._split_args(arg)
        layouts = self._parse_keyboard_names(args[0])

        if layouts is None or len(layouts) != 1:
            self._warn("usage: polish <keyboard>")
            return
        
        layout = layouts[0]
        # TODO: implement polishing
        self._warn("polish not implemented yet")



    def help_swap(self) -> None:
        self._command_help(
            command="swap",
            description="Swaps two positions on the keyboard, for those hand crafted, fine tuning adjustments to a layout.",
            argument_name_description=[
                ("keyboard", "<keyboard>", "the keyboard layout to swap positions on, can be a layout name or the index of a layout in memory."),
                ("position1", "<position1>", "the first position to swap, just name the character that is there."), 
                ("position2", "<position2>", "the second position to swap.")
            ],
            examples=["0 d h", "qwerty a q"],
        )
        
    def complete_swap(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        # if completing the keyboard, list all keyboard names
        if begidx == 0:
            return self._list_keyboard_names(text)

        return []

    def do_swap(self, arg: str) -> None:
        """editing: swaps two positions on the keyboard"""
        args = self._split_args(arg)
        if len(args) != 3:
            self._warn("usage: swap <keyboard> <position1> <position2>")
            return
        
        self._warn("swap not implemented yet")
        # TODO


    def help_mirror(self) -> None:
        self._command_help(
            command="mirror",
            description="Mirrors a keyboard layout horizontally, so that the left and right hands are swapped.",
            argument_name_description=[("keyboard", "<keyboard>", "the keyboard layout to mirror, can be a layout name or the index of a layout in memory.")],
            examples=["0", "hdpm"],
        )
        
    def complete_mirror(self, text: str, line: str, begidx: int, endidx: int) -> list[str]: # pyright: ignore[reportUnusedParameter]
        return self._list_keyboard_names(text)

    def do_mirror(self, arg: str) -> None:
        """editing: mirrors a keyboard layout horizontally"""
        args = self._split_args(arg)

        self._warn("mirror not implemented yet")
        # TODO


    def help_pin(self) -> None:
        self._command_help(
            command="pin",
            description="Pins characters to their current positions, so that they cannot be swapped or moved to a different position during editing (`generate`, `improve`, etc.). "
            "To see current pins, pass no arguments (`pin`). To clear all pins, use `pin nothing`.",
            argument_name_description=[
                ("nothing", "[nothing]", "to remove all existing pins, specify `pin nothing`."),
                ("char", "[<char> ...]", "the characters to pin, can be a single character or a space-separated list of characters.")
            ],
            examples=["", "a b c", "nothing"],
        )

    def do_pin(self, arg: str) -> None:
        """editing: pins characters to their current position"""
        args = self._split_args(arg)
        
        if not args:
            pass

        elif args[0].lower() in ('nothing', 'clear'):
            self.pinned_chars = []

        else:
            invalid_chars = [char for char in args if char not in self.model.freqdist.char_seq]
            if invalid_chars:
                str_invalid_chars = ' '.join(invalid_chars)
                self._warn(f"Warning: {len(invalid_chars)} character(s) are not in the corpus: {str_invalid_chars}")

            self.pinned_chars.extend(args)

        # display pinned characters
        if not self.pinned_chars:
            self._info("nothing pinned.")
        else:
            str_pinned_chars = ' '.join(self.pinned_chars)
            self._info(f"pinned: {str_pinned_chars}")



    def help_memory(self) -> None:
        self._command_help(
            command="memory",
            description="Shows the best N layouts in memory, that is, the lowest scores based on the current objective (see `help objective`).",
            argument_name_description=[("N", "[N=10]", "the number of layouts to show, default is 10.")],
            examples=["", "100"],
        )

    def do_memory(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        """editing: shows the best layouts that are loaded in memory"""
        args = self._split_args(arg)
        if len(args) > 0:
            try:
                N = int(args[0])
            except ValueError:
                self._warn("N must be an integer: memory [N=10]")
                return
        else:
            N = 10

        self._info(self._layout_memory_to_str(top_n=N))



    def help_reload(self) -> None:
        self._command_help(
            command="reload",
            description="Reload the settings from the config.toml file. Keeps generated layouts results in memory, but updates corpus, hardware, and objective function.",
            argument_name_description=[],
            examples=[""]
        )

    def do_reload(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        """configuration: reload the settings from the config.toml file"""
        # should reload the settings from the config.toml file
        self._load_settings()

        

    def help_objective(self) -> None:
        self._command_help(
            command="objective",
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
            argument_name_description=[("formula", "[<formula>]", "when specified, sets the objective function to the given formula.")],
            examples=["", "100sfb + 6effort + 60pinky_ring + 60scissors_ortho + 60sfs - 5alt"]
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
        """configuration: view or update the objective function used to score layouts"""
        
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



    def help_metrics(self) -> None:
        self._command_help(
            command="metrics",
            description="Shows all current metrics and their descriptions.",
            argument_name_description=[],
            examples=[""]
        )

    def do_metrics(self, arg: str) -> None: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        """configuration: shows metric names and descriptions"""
        header = ['Metric', 'Description']

        rows = []

        for metric in self.metrics:
            if metric.name in self.break_before_metrics:
                rows.append([None] * 2) 
            rows.append([metric.name, metric.description])

        self._info(tabulate(rows, headers=header, tablefmt="simple"))


    def help_save(self) -> None:
        self._command_help(
            command="save",
            description="Saves a layout to the ./layouts directory. Saved layouts can then be used in other commands by name.",
            argument_name_description=[
                ("keyboard", "<keyboard>", "the keyboard layout to save, can be a layout name or the index of a layout in memory."),
                ("name", "[<name>]", "the name of the new layout, defaults to the home row characters.")
            ],
            examples=["0", "1 mylayout"],
        )

    def do_save(self, arg: str) -> None:
        """editing: save new layouts from memory to a new file"""
        args = self._split_args(arg)
        layouts = self._parse_keyboard_names(arg[0])
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
            super().do_help(arg)
            return
        
        names = self.get_names()

        commands = []
        for name in names:
            if name[:3] != 'do_':
                continue
            
            try:
                doc=getattr(self, name).__doc__
                doc = cleandoc(doc)
            except AttributeError:
                doc = ''

            commands.append((name[3:], doc))

        # analyze  contributions  generate  improve  metrics    pin   reload
        # compare  exit           help      memory   objective  quit  save  

        headers = ['analysis', 'editing', 'configuration', 'commands']

        sections = {header: [] for header in headers}
        for name, doc in commands:
            for header in headers:
                if doc.startswith(header):
                    sections[header].append((name, doc[len(header):].strip(':').strip()))
                    break
            else:
                sections['commands'].append((name, doc))

        self._info("")
        self._info("Type `help <command>` to get help on a specific command.")
        self._info("")
        for header in headers:
            if not sections[header]:
                continue
            self._info(f"\n{header}:")
            self._info(textwrap.indent(
                tabulate(sections[header], tablefmt="plain", maxcolwidths = 70),
                "  "
            ))
            self._info("")


    def do_quit(self, arg: str) -> bool: # pyright: ignore[reportArgumentType, reportUnusedParameter]
        """quits Jalo"""
        self._info("Exiting Jalo REPL. Bye bye.")
        self._info("")
        return True

    def do_exit(self, arg: str) -> bool:
        """quits Jalo"""
        return self.do_quit(arg)


    # ----- helpers -------------------------------------------------------
    @staticmethod
    def _split_args(arg: str) -> List[str]:
        try:
            return shlex.split(arg)
        except ValueError as exc:
            print(f"error parsing arguments: {exc}", file=sys.stderr)
            return []

    def _info(self, message: str) -> None:
        print(message)

    @staticmethod
    def _warn(message: str) -> None:
        print(message, file=sys.stderr)
    

    def _list_keyboard_names(self, prefix: str = '') -> list[str]:        
        # read all file names in the ./layouts directory that start with prefix and end with .kb
        return [
            f.stem
            for f in Path('layouts').glob(f'{prefix}*.kb')
        ]


    def _parse_keyboard_names(self, arg: str) -> list[KeyboardLayout] | None:
        names = self._split_args(arg)
        if len(names) < 1:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return None

        layouts = []

        for name in names:
            layout = None
            
            # Try to interpret as an int index into self.layout_memory
            try:
                idx = int(name)
                if not self.layouts_memory:
                    self._warn(f"No layouts in memory, so cannot retrieve '{name}'. Use 'generate', 'improve', or retrieve by name from ./layouts/.")
                    return None
                if 0 <= idx < len(self.layouts_memory):
                    layouts.append(self.layouts_memory[idx])
                else:
                    self._warn(f"No layout found with index {idx} in memory of {len(self.layouts_memory)} layouts")
                    return None

            except ValueError:
                # Not an int, try loading by name

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

    def _layout_memory_to_str(self, original_score: float | None = None, top_n: int = 10) -> str:
        res = []
        for li,layout in enumerate(self.layouts_memory[:top_n]):
            score = self._get_model(layout).score_layout(layout)
            if original_score is None:
                original_score = score
            delta = score - original_score
            layout_str = f"layout {li} {score*100:.3f} {delta*100:.3f}\n"
            layout_str += f'{layout}\n'
            res.append(layout_str)
        
        return '\n'.join(res)

    def _layout_memory_from_optimizer(self, optimizer: Optimizer, original_layout: KeyboardLayout | None = None):
        self.layouts_memory = []
        for i, new_char_at_pos in enumerate(optimizer.population.sorted()[:10]):
            new_layout = optimizer.model.layout_from_char_at_positions(new_char_at_pos, original_layout=original_layout, name = f'{i}')
            self.layouts_memory.append(new_layout)


    def _tabulate_analysis(self, layouts: List[KeyboardLayout], show_contributions: bool = False) -> str:

        model_for_layout = {layout: self._get_model(layout) for layout in layouts}
        show_compare = len(layouts)>1
        show_top_ngram = len(layouts)==1

        try:
            analysis = {layout: model_for_layout[layout].analyze_layout(layout) for layout in layouts}
            scores = {layout: model_for_layout[layout].score_layout(layout) for layout in layouts}
            contributions = {layout: model_for_layout[layout].score_contributions(layout) for layout in layouts}
            top_ngram = {layout: model_for_layout[layout].top_ngram_per_metric(layout) for layout in layouts} if show_top_ngram else {layout: {} for layout in layouts}

        except Exception as e:
            self._warn(f"Error: could not analyze layout: {e}")
            return ''

        weights = {}

        def col_sel(cols):
            '''
            cols are:
            0: metric value
            1: comparison indicator
            2: contributions indicator
            3: top n-gram indicator
            '''
            res = [cols[0]]

            if show_compare:
                res.append(cols[1])
            
            if show_contributions:
                res.append(cols[2])

            if show_top_ngram:
                res.append(cols[3])

            return res


        def format_ngrams(layout: KeyboardLayout, ngram_values: dict[tuple[int, ...], float], max_length: int = 50) -> str:
            SEP = ', '
            res = []
            cum_len = 0

            for ngram, value in sorted(ngram_values.items(), key=lambda x: x[1], reverse=True):
                chars = ''.join(
                    layout.char_at_position[layout.hardware.positions[pi]]
                    for pi in ngram
                )
                item = f"{chars}: {100*value:.3f}"

                if cum_len + len(item) > max_length:
                    break
                res.append(item)
                cum_len += len(item) + len(SEP)

            return SEP.join(res)

        top_ngram_str = {
            layout: {
                metric: format_ngrams(layout, top_ngram[layout][metric]) if layout in top_ngram and metric in top_ngram[layout] else None
                for metric in self.metrics
            }
            for layout in layouts
        } 

        if show_contributions:
            header = ['metric', 'w']
        else:
            header = ['metric']

        header.extend([item for layout in layouts for item in col_sel([f"{layout.name}\n{layout.hardware.name}", 'Δ', 'ΔS', 'top n-gram'])])
        
        rows = []
        for metric in self.metrics:
            minv = min(analysis[layout][metric] for layout in layouts)
            maxv = max(analysis[layout][metric] for layout in layouts)
            delta = maxv - minv

            def delta_sign(v: float) -> str: 
                return (
                    '' if (maxv<0.003 or delta/maxv < 0.1 or delta < 0.002) else 
                    '+' if maxv-v < delta/10 else 
                    '-' if v-minv < delta/10 else 
                    ''
                )
            

            if metric.name in self.break_before_metrics:
                rows.append([None] * (len(layouts) * 2 + 1))
            rows.append(
                ([metric.name, self.model.objective.metrics.get(metric, None)] if show_contributions else [metric.name]) + 
                [
                    item 
                    for layout in layouts 
                    for item in col_sel([
                        analysis[layout][metric]*100 if metric in analysis[layout] else None, 
                        delta_sign(analysis[layout][metric]),
                        contributions[layout][metric]*100 if metric in contributions[layout] and abs(contributions[layout][metric]) > 0.01 else None,
                        top_ngram_str[layout][metric]
                    ])
                ]
            )

        rows.append([None] * (len(layouts) * 2 + 1))
        rows.append((['score', None] if show_contributions else ['score']) + [
            item 
            for layout in layouts 
            for item in (
                col_sel([None, None, f"{scores[layout]*100:.3f}", None]) if show_contributions else 
                col_sel([f"{scores[layout]*100:.3f}", None, None, None])
            )
        ])
        
        return tabulate(rows, headers=header, tablefmt="simple", floatfmt=".3f")



    def _command_help(self, command: str, description: str, argument_name_description: list[tuple[str, str, str]], examples: list[str]) -> None:
        args_usage_str = ' '.join([f"{syntax}" for _, syntax, _ in argument_name_description])
        args_lines = '\n'.join([textwrap.fill(f"{arg_name}: {description}", width=80, initial_indent="  ", subsequent_indent="    ") for arg_name, _, description in argument_name_description])

        # reflow description: make sure all lines are at most 80 characters
        description = '\n'.join([textwrap.fill(line, width=80) for line in description.split('\n')])

        self._info("\n".join([
            "",
            f"usage: {command} {args_usage_str}",
            "",
            description,
            "",
            "Arguments:",
            args_lines,
            "",
            "Examples:",
            '\n'.join([f"  {command} {example}" for example in examples]),
            ""
        ]))


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

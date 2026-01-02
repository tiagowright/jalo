"""Core Jalo shell implementation and shared helpers."""



import cmd
import dataclasses
import importlib
import pkgutil
import re
import shlex
import sys
import types
from inspect import cleandoc
from pathlib import Path
from typing import Callable, List, Optional

from layout import KeyboardLayout
from model import KeyboardModel
from freqdist import FreqDist
from metrics import METRICS, use_oxeylyzer_mode
from objective import ObjectiveFunction
from hardware import KeyboardHardware
from optim import Optimizer
from repl.memory import LayoutMemoryManager
from repl.formatting import (
    format_analysis_table,
    format_command_help,
    format_help_summary,
    format_intro,
    format_layout_memory,
    format_settings_info,
)


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


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
    category: str
    short_description: str


CommandHandler = Callable[["JaloShell", str], Optional[bool]]
CommandCompleter = Callable[["JaloShell", str, str, int, int], List[str]]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_CONFIG = """
hardware = "ortho"
corpus = "en"
"""


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
        return cls(
            hardware=str(hardware),
            corpus=str(corpus),
            oxeylyzer_mode=bool(oxeylyzer_mode),
            layouts_memory_size=int(layouts_memory_size),
            objective=str(objective),
        )


class JaloShell(cmd.Cmd):
    """Interactive shell for exploring keyboard layouts."""

    prompt = "jalo> "
    break_before_metrics = [
        "home",
        "heat",
        "lsb",
        "same_hand",
        "roll",
        "redirect",
        "left_hand",
        "finger_0",
        "sfb_finger_0",
    ]

    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path or DEFAULT_CONFIG_PATH

        self.memory = LayoutMemoryManager()
        self.model_for_hardware: dict[KeyboardHardware, KeyboardModel] = {}
        self.current_command = ""
        self.commands: dict[str, Command] = {}

        self._load_settings()
        self._register_builtin_commands()

    # ----- lifecycle hooks -------------------------------------------------
    def preloop(self) -> None:
        self._info(format_intro(self._load_settings_str()))

    def precmd(self, line: str) -> str:
        """Called before each command is executed. Save the command line."""

        self.current_command = line.strip()
        line = re.sub(r"(?<!\\)#.*", "", line)
        return line

    def onecmd(self, line: str) -> bool:
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            self._info("")
            self._info("Command interrupted.\n")
            return False

    def script(self, script: str):
        """Parse and execute the script as a sequence of commands"""
        for line in script.splitlines():

            line = line.strip()
            line = self.precmd(line)
            if not line:
                continue
            
            stop = self.onecmd(line)
            stop = self.postcmd(stop, line)
            self._info("")

            if stop:
                break


    # ----- command registration -------------------------------------------
    def register_command(
        self,
        command: Command,
        handler: CommandHandler,
        completer: CommandCompleter | None = None,
    ) -> None:
        self.commands[command.name] = command
        setattr(self, f"do_{command.name}", types.MethodType(handler, self))
        if completer is not None:
            setattr(self, f"complete_{command.name}", types.MethodType(completer, self))

    def _register_builtin_commands(self) -> None:
        from repl import cmds as cmds_pkg

        package_path = Path(cmds_pkg.__file__).parent
        for module_info in pkgutil.iter_modules([str(package_path)]):
            if module_info.ispkg:
                continue

            module_name = module_info.name
            module = importlib.import_module(f"{cmds_pkg.__name__}.{module_name}")

            exec_fn = getattr(module, "exec", None)
            if exec_fn is None:
                continue

            desc_fn = getattr(module, "desc", None)
            if callable(desc_fn):
                command = desc_fn()
            else:
                command = Command(
                    name=module_name,
                    description="",
                    arguments=(),
                    examples=(),
                    category="commands",
                    short_description="",
                )

            self._bind_command(command, exec_fn, getattr(module, "complete", None))  # pyright: ignore[reportArgumentType]

    def _bind_command(
        self,
        command: Command,
        handler_fn: Callable[["JaloShell", str], Optional[bool]],
        completer_fn: Callable[["JaloShell", str, str, int, int], List[str]] | None = None,
    ) -> None:
        def handler(self: "JaloShell", arg: str, _fn=handler_fn) -> Optional[bool]:
            return _fn(self, arg)

        setattr(self, f"do_{command.name}", types.MethodType(handler, self))
        self.commands[command.name] = command

        if completer_fn is None:
            return

        def completer(
            self: "JaloShell", text: str, line: str, begidx: int, endidx: int, _fn=completer_fn
        ) -> List[str]:
            return _fn(self, text, line, begidx, endidx)

        setattr(self, f"complete_{command.name}", types.MethodType(completer, self))

    def get_names(self) -> list[str]:
        return dir(self)

    # ----- configuration helpers ------------------------------------------
    def _load_settings(self) -> None:
        self.settings = _load_settings(self.config_path)
        
        self.metrics = METRICS
        use_oxeylyzer_mode(self.settings.oxeylyzer_mode)
        
        self._change_settings(
            objective = ObjectiveFunction.from_name(self.settings.objective),
            hardware = KeyboardHardware.from_name(self.settings.hardware),
            freqdist = FreqDist.from_name(self.settings.corpus)
        )

        self.pinned_chars: list[str] = []

    def _load_settings_str(self) -> str:
        return format_settings_info(
            self.config_path,
            self.hardware.name,
            self.freqdist.corpus_name,
            str(self.objective),
        )

    def _change_settings(self, objective: ObjectiveFunction | None = None, hardware: KeyboardHardware | None = None, freqdist: FreqDist | None = None):
        if objective is None and hardware is None and freqdist is None:
            return
        
        self.freqdist = freqdist or self.freqdist
        self.hardware = hardware or self.hardware
        self.objective = objective or self.objective
        
        self.model = KeyboardModel(
            hardware=self.hardware,
            metrics=self.metrics,
            objective=self.objective,
            freqdist=self.freqdist,
        )
        self.model_for_hardware = {self.hardware: self.model}


    def _get_model(self, layout: KeyboardLayout) -> KeyboardModel:
        if layout.hardware not in self.model_for_hardware:
            self.model_for_hardware[layout.hardware] = KeyboardModel(
                hardware=layout.hardware,
                metrics=self.metrics,
                objective=self.objective,
                freqdist=self.freqdist,
            )
        return self.model_for_hardware[layout.hardware]

    # ----- message helpers -------------------------------------------------
    def _info(self, message: str) -> None:
        print(message)

    @staticmethod
    def _warn(message: str) -> None:
        print(message, file=sys.stderr)

    # ----- parsing helpers ------------------------------------------------
    @staticmethod
    def _split_args(arg: str) -> List[str]:
        try:
            return shlex.split(arg)
        except ValueError as exc:
            print(f"error parsing arguments: {exc}", file=sys.stderr)
            return []

    def _arg_num_at_index(self, line: str, begidx: int, endidx: int) -> int | None:
        try:
            args = shlex.split(line[:begidx])
        except ValueError:
            return None
        return len(args)

    # ----- memory helpers -------------------------------------------------
    def _layout_memory_to_str(
        self, list_num: int | None = None, top_n: int = 10
    ) -> str:
        def score_fn(layout: KeyboardLayout) -> float:
            return self._get_model(layout).score_layout(layout)

        return format_layout_memory(self.memory, list_num, top_n, score_fn)

    def _layout_memory_from_optimizer(
        self,
        optimizer: Optimizer,
        original_layout: KeyboardLayout | None = None,
        push_to_stack: bool = False,
    ) -> int | None:
        layouts = []
        for new_char_at_pos in optimizer.population.sorted()[:10]:
            new_layout = optimizer.model.layout_from_char_at_positions(
                new_char_at_pos, original_layout=original_layout, name=""
            )
            layouts.append(new_layout)

        if push_to_stack:
            for layout in layouts:
                self.memory.push_to_stack(layout, self.current_command, original_layout)
            return None
        return self.memory.add_list(layouts, self.current_command, original_layout)

    def _push_layout_to_stack(
        self, layout: KeyboardLayout, original_layout: KeyboardLayout | None = None
    ) -> None:
        self.memory.push_to_stack(layout, self.current_command, original_layout)

    def _tabulate_analysis(
        self, layouts: List[KeyboardLayout], show_contributions: bool = False
    ) -> str:
        model_for_layout = {layout: self._get_model(layout) for layout in layouts}
        show_top_ngram = len(layouts) == 1

        try:
            analysis = {
                layout: model_for_layout[layout].analyze_layout(layout)
                for layout in layouts
            }
            scores = {
                layout: model_for_layout[layout].score_layout(layout)
                for layout in layouts
            }
            contributions = (
                {layout: model_for_layout[layout].score_contributions(layout) for layout in layouts}
                if show_contributions
                else None
            )
            top_ngram = (
                {layout: model_for_layout[layout].top_ngram_per_metric(layout) for layout in layouts}
                if show_top_ngram
                else None
            )
        except Exception as e:
            self._warn(f"Error: could not analyze layout: {e}")
            return ""

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

    # ----- core shell commands -------------------------------------------
    def do_help(self, arg: str) -> None:  # type: ignore[override]
        """help with any command, try `help analyze`"""

        if arg:
            if arg in self.commands:
                cmd_meta = self.commands[arg]
                argument_name_description = [
                    (argument.name, argument.syntax, argument.description)
                    for argument in cmd_meta.arguments
                ]
                self._info(
                    format_command_help(
                        command=cmd_meta.name,
                        description=cmd_meta.description,
                        argument_name_description=argument_name_description,
                        examples=list(cmd_meta.examples),
                    )
                )
            else:
                super().do_help(arg)
            return

        names = self.get_names()
        headers = ["analysis", "optimization", "editing", "configuration", "commands"]
        commands_list: dict[str, list[tuple[str, str]]] = {header: [] for header in headers}

        for name in names:
            if not name.startswith("do_"):
                continue
            cmd_name = name[3:]
            if cmd_name in self.commands:
                cmd_meta = self.commands[cmd_name]
                if cmd_meta.category not in commands_list:
                    commands_list[cmd_meta.category] = []
                    headers.append(cmd_meta.category)
                commands_list[cmd_meta.category].append(
                    (cmd_name, cmd_meta.short_description)
                )
            else:
                try:
                    doc = getattr(self, name).__doc__
                    doc = cleandoc(doc)
                except AttributeError:
                    doc = ""
                commands_list["commands"].append((cmd_name, doc))

        self._info(format_help_summary(headers, commands_list))


def _load_settings(config_path: Path) -> JaloSettings:
    if not config_path.exists():
        print(
            f"warning: config file not found at {config_path}, creating default config",
            file=sys.stderr,
        )
        try:
            config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO errors should not kill REPL
            print(
                f"Fatal error: failed to create config {config_path}: {exc}",
                file=sys.stderr,
            )
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


# def main(argv: List[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(
#         description="Jalo keyboard layout analyzer and optimizer."
#     )
#     parser.add_argument(
#         "--no-history",
#         action="store_true",
#         help="Disable readline history persistence.",
#     )
#     args = parser.parse_args(argv)

#     if not args.no_history:
#         _configure_readline()



__all__ = [
    "Command",
    "CommandArgument",
    "JaloShell",
    "JaloSettings",
    "_configure_readline",
    "DEFAULT_CONFIG_PATH",
]

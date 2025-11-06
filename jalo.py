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
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate
from textwrap import dedent

from layout import KeyboardLayout
from model import KeyboardModel
from freqdist import FreqDist
from metrics import METRICS, Metric, ObjectiveFunction
from hardware import KeyboardHardware



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

    @classmethod
    def from_dict(cls, data: dict) -> "JaloSettings":
        hardware = data.get("hardware", "ortho")
        corpus = data.get("corpus", "en")
        return cls(hardware=str(hardware), corpus=str(corpus))


class JaloShell(cmd.Cmd):
    """Interactive shell for exploring keyboard layouts."""

    prompt = "jalo> "
    intro = "Jalo REPL – type 'help' to list commands."

    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path or Path(__file__).resolve().with_name("config.toml")
        self._load_settings()

    def _load_settings(self):
        self.settings = _load_settings(self.config_path)
        self.freqdist = FreqDist.from_name(self.settings.corpus)
        self.metrics = METRICS
        self.hardware = KeyboardHardware.from_name(self.settings.hardware)
        self.objective = ObjectiveFunction({self.metrics[0]: 2.0, self.metrics[2]: 1.5, self.metrics[3]: 3.0, self.metrics[18]: 1.1})
        self.model = KeyboardModel(hardware=self.hardware, metrics=self.metrics, objective=self.objective, freqdist=self.freqdist)


    # ----- core commands -------------------------------------------------
    def do_analyze(self, arg: str) -> None:
        """analyze <keyboard> - Analyze the given keyboard layout."""

        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: analyze <keyboard>")
            return

        self._info(self._tabulate_analysis(layouts))

    def do_contributions(self, arg: str) -> None:
        """contributions <keyboard> [<keyboard>...]"""
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: contributions <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts, show_contributions=True))

    def do_compare(self, arg: str) -> None:
        """compare <keyboard> [<keyboard>...]"""
        layouts = self._parse_keyboard_names(arg)
        if layouts is None:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return

        self._info(self._tabulate_analysis(layouts))
        
    def do_generate(self, arg: str) -> None:
        """generate"""
        self._info("[generate] placeholder layout generation.")

    def do_improve(self, arg: str) -> None:
        """improve"""
        self._info("[improve] placeholder layout improvement.")

    def do_reload(self, arg: str) -> None:
        """Reload the settings from the config.toml file. Keeps generated layouts results in memory, but updates corpus, hardware, and objective function."""
        # should reload the settings from the config.toml file
        self._load_settings()

        self._info(
            f"loaded settings from {self.config_path}:\n"
            f"hardware='{self.settings.hardware}' "
            f"corpus='{self.settings.corpus}'."
        )
    
    def do_objective(self, arg: str) -> None:
        """shows the current objective function"""
        self._info(f"Objective function: {self.objective}")


    # ----- shell controls ------------------------------------------------
    def do_help(self, arg: str) -> None:  # type: ignore[override]
        """help [command]"""
        super().do_help(arg)

    def do_quit(self, arg: str) -> bool:
        """quit"""
        self._info("Exiting Jalo REPL.")
        return True

    def do_exit(self, arg: str) -> bool:
        """exit"""
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
    
    def _parse_keyboard_names(self, arg: str) -> List[KeyboardLayout]:
        names = self._split_args(arg)
        if len(names) < 1:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return None

        try:
            layouts = [KeyboardLayout.from_name(name, self.hardware) for name in names]
        except FileNotFoundError as e:
            self._warn(f"could not find layout in: {e.filename}")
            return None
        
        return layouts


    def _tabulate_analysis(self, layouts: List[KeyboardLayout], show_contributions: bool = False) -> str:

        analysis = {layout: self.model.analyze_layout(layout) for layout in layouts}
        scores = {layout: self.model.score_layout(layout) for layout in layouts}
        contributions = {layout: self.model.score_contributions(layout) for layout in layouts}
    
        break_before = ['alt','left_hand','finger_0','sfb_finger_0']

        def col_sel(cols):
            return cols[:2] if not show_contributions else cols

        header = ['metric'] + [item for layout in layouts for item in col_sel([layout.name, 'Δ', 'ΔS'])]
        
        rows = []
        for metric in self.metrics:
            minv = min(analysis[layout][metric] for layout in layouts)
            maxv = max(analysis[layout][metric] for layout in layouts)
            delta = maxv - minv

            def delta_sign(v: float) -> str: 
                return (
                    '' if (minv<0.02 or delta/minv < 0.1) and delta < 0.02 else 
                    '+' if maxv-v < delta/10 else 
                    '-' if v-minv < delta/10 else 
                    ''
                )
            

            if metric.name in break_before:
                rows.append([None] * (len(layouts) * 2 + 1))
            rows.append(
                [metric.name] + 
                [
                    item 
                    for layout in layouts 
                    for item in col_sel([
                        analysis[layout][metric]*100 if metric in analysis[layout] else None, 
                        delta_sign(analysis[layout][metric]),
                        contributions[layout][metric]*100 if metric in contributions[layout] and abs(contributions[layout][metric]) > 0.01 else None
                    ])
                ]
            )

        rows.append([None] * (len(layouts) * 2 + 1))
        rows.append(['score'] + [
            item 
            for layout in layouts 
            for item in (
                col_sel([None, None, f"{scores[layout]*100:.3f}"]) if show_contributions else 
                col_sel([f"{scores[layout]*100:.3f}", None, None])
            )
        ])
        
        return tabulate(rows, headers=header, tablefmt="simple", floatfmt=".3f")




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
        description="Launch the Jalo interactive shell."
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable readline history persistence.",
    )
    args = parser.parse_args(argv)

    if not args.no_history:
        _configure_readline()

    config_path = Path(__file__).resolve().with_name("config.toml")
    shell = JaloShell(config_path=config_path)

    shell._info(
        f"Loaded config: hardware='{shell.settings.hardware}', "
        f"corpus='{shell.settings.corpus}'."
    )
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        shell._info("")  # ensure a newline after Ctrl+C
        shell._info("Interrupted. Bye bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

from layout import KeyboardLayout
from model import KeyboardModel
from freqdist import FreqDist
from metrics import METRICS, Metric
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

    def __init__(self, settings: Optional[JaloSettings] = None, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.config_path = config_path or Path(__file__).resolve().with_name("config.toml")
        self.settings = settings or _load_settings(self.config_path) or JaloSettings("ortho", "en")
        
        self.freqdist = FreqDist.from_name(self.settings.corpus)
        self.metrics = METRICS
        self.hardware = KeyboardHardware.from_name(self.settings.hardware)
        self.model = KeyboardModel(hardware=self.hardware, metrics=self.metrics, freqdist=self.freqdist)


    # ----- core commands -------------------------------------------------
    def do_analyze(self, arg: str) -> None:
        """
        analyze <keyboard>

        Analyze the given keyboard layout. Currently prints a placeholder.
        """
        keyboard = arg.strip()
        if not keyboard:
            self._warn("usage: analyze <keyboard>")
            return
        
        try:
            layout = KeyboardLayout.from_name(keyboard, self.hardware)
        except FileNotFoundError as e:
            self._warn(f"could not find layout in: {e.filename}")
            return

        analysis = self.model.analyze_layout(layout)
        self._info(self._tabulate_analysis([layout], {layout: analysis}))

    def do_compare(self, arg: str) -> None:
        """
        compare <keyboard> [<keyboard>...]

        Compare one or more keyboard layouts.
        """
        names = self._split_args(arg)
        if len(names) < 1:
            self._warn("usage: compare <keyboard> [<keyboard>...]")
            return

        try:
            layouts = [KeyboardLayout.from_name(name, self.hardware) for name in names]
        except FileNotFoundError as e:
            self._warn(f"could not find layout in: {e.filename}")
            return

        analysis = {layout: self.model.analyze_layout(layout) for layout in layouts}
        self._info(self._tabulate_analysis(layouts, analysis))
        
    def do_generate(self, arg: str) -> None:
        """generate"""
        self._info("[generate] placeholder layout generation.")

    def do_improve(self, arg: str) -> None:
        """improve"""
        self._info("[improve] placeholder layout improvement.")

    def do_reload(self, arg: str) -> None:
        """reload"""
        # should reload the settings from the config.toml file
        self.settings = _load_settings(Path(__file__).resolve().with_name("config.toml"))
        try:
            rel_path = self.config_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = self.config_path
        self._info(
            f"[reload] loaded settings from {rel_path}: "
            f"hardware='{self.settings.hardware}' "
            f"corpus='{self.settings.corpus}'."
        )

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



    def _tabulate_analysis(self, layouts: List[KeyboardLayout], analysis: dict[KeyboardLayout, dict[Metric, float]]) -> str:
        break_before = ['alt','left_hand','finger_0','sfb_finger_0']

        # sort metrics by name
        header = ['metric'] + [item for layout in layouts for item in [layout.name, 'Δ']]
        
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
                    for item in [
                        analysis[layout][metric]*100 if metric in analysis[layout] else '', 
                        delta_sign(analysis[layout][metric])
                    ]
                ]
            )
        return tabulate(rows, headers=header, tablefmt="simple", floatfmt=".3f")


def _load_settings(config_path: Path) -> JaloSettings:
    if not config_path.exists():
        return JaloSettings("ortho", "en")

    try:
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:  # pragma: no cover - IO errors should not kill REPL
        print(f"warning: failed to read config {config_path}: {exc}", file=sys.stderr)
        return JaloSettings("ortho", "en")

    if not isinstance(data, dict):
        print(f"warning: malformed config in {config_path}", file=sys.stderr)
        return JaloSettings("ortho", "en")

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
    settings = _load_settings(config_path)

    shell = JaloShell(settings=settings, config_path=config_path)
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

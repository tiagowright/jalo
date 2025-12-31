"""`reload` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.shell import Command

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="reload",
        description="Reload the settings from the config.toml file. Keeps generated layouts results in memory, but updates corpus, hardware, and objective function.",
        arguments=(),
        examples=("",),
        category="commands",
        short_description="reload the settings from the config.toml file",
    )


def exec(shell: "JaloShell", arg: str) -> None:
    shell._load_settings()


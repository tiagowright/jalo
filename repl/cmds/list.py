"""`list` command for the Jalo shell."""



from typing import TYPE_CHECKING

from repl.shell import Command, CommandArgument

if TYPE_CHECKING:  # pragma: no cover
    from repl.shell import JaloShell


def desc() -> Command:
    return Command(
        name="list",
        description=(
            "Lists layouts in memory. Called with no arguments, shows available lists and top 3 layouts from the stack. "
            "Called with a list number, shows layouts from that list. Called with two arguments, the second is the number of layouts to show."
        ),
        arguments=(
            CommandArgument(
                "list_num",
                "[<list_num>]",
                "the list number to show (0 for stack, >0 for numbered lists). If omitted, shows all lists and top 3 from stack.",
            ),
            CommandArgument(
                "count",
                "[<count>=10]",
                "the number of layouts to show, default is 10 for numbered lists, 3 for stack when no arguments.",
            ),
        ),
        examples=("", "0", "1", "2 5"),
        category="editing",
        short_description="lists layouts in memory",
    )


def exec(shell: "JaloShell", arg: str) -> None:
    args = shell._split_args(arg)

    if len(args) == 0:
        shell._info("Available lists:")
        if len(shell.memory.stack) > 0:
            shell._info(f"  stack: {len(shell.memory.stack)} layouts")
        else:
            shell._info("  stack: (empty)")

        list_numbers = shell.memory.get_all_list_numbers()
        if list_numbers:
            for list_num in list_numbers:
                if list_num in shell.memory.lists:
                    count = len(shell.memory.lists[list_num].layouts)
                    command_name = shell.memory.lists[list_num].command
                    shell._info(f"  {list_num}: {count} layouts  > {command_name}")
        else:
            shell._info("  (no numbered lists)")

        shell._info("")
        shell._info("Top 3 layouts from stack:")
        shell._info(shell._layout_memory_to_str(list_num=0, top_n=3))
        return

    try:
        list_num = int(args[0])
    except ValueError:
        shell._warn("list number must be an integer: list [<list_num>] [<count>]")
        return

    if len(args) > 1:
        try:
            count = int(args[1])
        except ValueError:
            shell._warn("count must be an integer: list [<list_num>] [<count>]")
            return
    else:
        count = 10 if list_num > 0 else 3

    shell._info(shell._layout_memory_to_str(list_num=list_num, top_n=count))


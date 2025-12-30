"""Text formatting and UI utilities for the Jalo keyboard tooling.

This module provides flexible functions for formatting text output, tables,
and various UI elements used throughout the Jalo REPL.
"""

import textwrap
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tabulate import tabulate

from layout import KeyboardLayout
from repl.memory import LayoutMemoryManager


def format_table(
    rows: list[list[Any]],
    headers: Optional[list[str]] = None,
    tablefmt: str = "simple",
    floatfmt: str = ".3f",
    disable_numparse: bool = False,
    maxcolwidths: Optional[list[int]] = None,
) -> str:
    """Format data as a table using tabulate.
    
    Args:
        rows: List of rows, where each row is a list of values
        headers: Optional list of header strings
        tablefmt: Table format (e.g., "simple", "plain")
        floatfmt: Format string for floating point numbers
        disable_numparse: Whether to disable number parsing
        maxcolwidths: Optional list of max column widths
        
    Returns:
        Formatted table string
    """
    kwargs = {
        "tablefmt": tablefmt,
        "floatfmt": floatfmt,
        "disable_numparse": disable_numparse,
    }
    if maxcolwidths is not None:
        kwargs["maxcolwidths"] = maxcolwidths
    if headers is not None:
        kwargs["headers"] = headers
    
    return tabulate(rows, **kwargs)


def format_layout_display(layout: KeyboardLayout) -> str:
    """Format a keyboard layout with its hardware information side by side.
    
    Args:
        layout: The keyboard layout to display
        
    Returns:
        Formatted string showing layout and hardware side by side
    """
    header = [layout.name, layout.hardware.name]
    layout_str = str(layout)
    hardware_str = layout.hardware.str(show_finger_numbers=True, show_stagger=True)
    
    # Tabulate removes leading spaces, so add a leading character to preserve formatting
    LEAD_SPACE = "| "
    rows = zip(
        [LEAD_SPACE + line for line in layout_str.split('\n')],
        [LEAD_SPACE + line for line in hardware_str.split('\n')]
    )
    
    return format_table(list(rows), headers=header, tablefmt="simple", disable_numparse=True)  # pyright: ignore[reportArgumentType]


def format_ngrams(
    layout: KeyboardLayout,
    ngram_values: dict[tuple[int, ...], float],
    max_length: int = 50,
    sep: str = ", ",
) -> str:
    """Format n-gram values as a compact string.
    
    Args:
        layout: The keyboard layout
        ngram_values: Dictionary mapping position tuples to values
        max_length: Maximum length of the formatted string
        sep: Separator between n-gram items
        
    Returns:
        Formatted string of n-grams
    """
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
        cum_len += len(item) + len(sep)
    
    return sep.join(res)


def format_analysis_table(
    layouts: Sequence[KeyboardLayout],
    metrics: Sequence[Any],
    analysis: dict[KeyboardLayout, dict[Any, float]],
    scores: dict[KeyboardLayout, float],
    contributions: Optional[dict[KeyboardLayout, dict[Any, float]]] = None,
    top_ngram: Optional[dict[KeyboardLayout, dict[Any, dict[tuple[int, ...], float]]]] = None,
    objective_weights: Optional[dict[Any, float]] = None,
    break_before_metrics: Optional[set[str]] = None,
    show_contributions: bool = False,
) -> str:
    """Format analysis results as a table comparing layouts across metrics.
    
    Args:
        layouts: List of layouts to compare
        metrics: List of metric objects
        analysis: Dictionary mapping layouts to metric values
        scores: Dictionary mapping layouts to total scores
        contributions: Optional dictionary mapping layouts to metric contributions
        top_ngram: Optional dictionary mapping layouts to top n-grams per metric
        objective_weights: Optional dictionary mapping metrics to weights
        break_before_metrics: Optional set of metric names that should have a blank row before them
        show_contributions: Whether to show contribution columns
        
    Returns:
        Formatted analysis table string
    """
    show_compare = len(layouts) > 1
    show_top_ngram = len(layouts) == 1 and top_ngram is not None
    break_before_metrics = break_before_metrics or set()
    
    def col_sel(cols: list[Any]) -> list[Any]:
        """Select columns based on what's being shown."""
        res = [cols[0]]  # metric value
        if show_compare:
            res.append(cols[1])  # comparison indicator
        if show_contributions:
            res.append(cols[2])  # contributions
        if show_top_ngram:
            res.append(cols[3])  # top n-grams
        return res
    
    # Format top n-grams if needed
    top_ngram_str: dict[KeyboardLayout, dict[Any, Optional[str]]] = {}
    if show_top_ngram and top_ngram:
        top_ngram_str = {
            layout: {
                metric: format_ngrams(layout, top_ngram[layout][metric])
                    if layout in top_ngram and metric in top_ngram[layout] else None
                for metric in metrics
            }
            for layout in layouts
        }
    else:
        top_ngram_str = {layout: {metric: None for metric in metrics} for layout in layouts}
    
    # Build header
    if show_contributions:
        header = ['metric', 'w']
    else:
        header = ['metric']
    
    header.extend([
        item for layout in layouts
        for item in col_sel([
            f"{layout.name}\n{layout.hardware.name}",
            'Δ',
            'ΔS',
            'top n-gram'
        ])
    ])
    
    # Build rows
    rows = []
    for metric in metrics:
        if metric.name in break_before_metrics:
            rows.append([None] * len(header))
        
        minv = min(analysis[layout][metric] for layout in layouts)
        maxv = max(analysis[layout][metric] for layout in layouts)
        delta = maxv - minv
        
        def delta_sign(v: float) -> str:
            """Determine comparison indicator."""
            if maxv < 0.003 or delta / maxv < 0.1 or delta < 0.002:
                return ''
            if maxv - v < delta / 10:
                return '+'
            if v - minv < delta / 10:
                return '-'
            return ''
        
        row = []
        if show_contributions:
            row.append(metric.name)
            row.append(objective_weights.get(metric, None) if objective_weights else None)
        else:
            row.append(metric.name)
        
        for layout in layouts:
            metric_value = analysis[layout][metric] * 100 if metric in analysis[layout] else None
            delta_indicator = delta_sign(analysis[layout][metric]) if show_compare else None
            contribution_value = None
            if show_contributions and contributions and layout in contributions:
                contrib = contributions[layout].get(metric, 0)
                contribution_value = contrib * 100 if abs(contrib) > 0.01 else None
            ngram_str = top_ngram_str[layout][metric] if layout in top_ngram_str else None
            
            row.extend(col_sel([metric_value, delta_indicator, contribution_value, ngram_str]))
        
        rows.append(row)
    
    # Add score row
    rows.append([None] * len(header))
    score_row = []
    if show_contributions:
        score_row.extend(['score', None])
    else:
        score_row.append('score')
    
    for layout in layouts:
        score_value = f"{scores[layout]*100:.3f}"
        if show_contributions:
            score_row.extend(col_sel([None, None, score_value, None]))
        else:
            score_row.extend(col_sel([score_value, None, None, None]))
    
    rows.append(score_row)
    
    return format_table(rows, headers=header, tablefmt="simple", floatfmt=".3f")


def format_layout_memory(
    memory: LayoutMemoryManager,
    list_num: Optional[int],
    top_n: int,
    score_fn: Callable[[KeyboardLayout], float],
) -> str:
    """Format layouts from memory as a string.
    
    Args:
        memory: The layout memory manager
        list_num: List number (None or 0 for stack, >0 for numbered list)
        top_n: Number of layouts to show
        score_fn: Function to compute layout scores
        
    Returns:
        Formatted string of layouts
    """
    if list_num is None or list_num == 0:
        # Show stack
        if len(memory.stack) == 0:
            return "No layouts found."
        
        layouts = memory.stack
        top_n_actual = min(top_n, len(layouts))
        items_to_show = layouts[-top_n_actual:]
        metadata_to_show = memory.stack_metadata[-top_n_actual:]
        
        res = []
        for li, (layout, metadata) in enumerate(zip(items_to_show, metadata_to_show)):
            score = score_fn(layout)
            original_score = score_fn(metadata.original_layout) if metadata.original_layout else None
            delta_str = f" ({(score - original_score)*100:.3f})" if original_score is not None else ''
            
            layout_str = f"layout {layout.name} {score*100:.3f}{delta_str}  > {metadata.command}\n"
            layout_str += f'{layout}\n'
            res.append(layout_str)
        
        return '\n'.join(res)
    else:
        # Show numbered list
        if list_num not in memory.lists:
            return f"No list {list_num} found."
        
        layouts = memory.lists[list_num].layouts
        if not layouts:
            return "No layouts found."
        
        top_n_actual = min(top_n, len(layouts))
        layouts_to_show = layouts[:top_n_actual]
        
        command = memory.lists[list_num].command or ""
        res = [f"layout list {list_num} > {command}", ""]
        original_layout = memory.lists[list_num].original_layout
        
        for li, layout in enumerate(layouts_to_show):
            score = score_fn(layout)
            original_score = score_fn(original_layout) if original_layout else None
            delta_str = f" ({(score - original_score)*100:.3f})" if original_score is not None else ''
            
            layout_num = f"{list_num}.{li + 1}"  # Numbered: 1.1, 1.2, ...
            layout_str = f"layout {layout_num} {score*100:.3f}{delta_str}\n"
            layout_str += f'{layout}\n'
            res.append(layout_str)
        
        return '\n'.join(res)


def format_command_help(
    command: str,
    description: str,
    argument_name_description: list[tuple[str, str, str]],
    examples: list[str],
    width: int = 80,
) -> str:
    """Format command help text.
    
    Args:
        command: Command name
        description: Command description (can be multi-line)
        argument_name_description: List of (arg_name, syntax, description) tuples
        examples: List of example usage strings
        width: Maximum line width
        
    Returns:
        Formatted help text
    """
    args_usage_str = ' '.join([syntax for _, syntax, _ in argument_name_description])
    args_lines = '\n'.join([
        textwrap.fill(f"{arg_name}: {desc}", width=width, initial_indent="  ", subsequent_indent="    ")
        for arg_name, _, desc in argument_name_description
    ]) if argument_name_description else "  none"
    
    # Reflow description: make sure all lines are at most width characters
    description = '\n'.join([
        textwrap.fill(line, width=width) for line in description.split('\n')
    ])
    
    return "\n".join([
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
    ])


def format_settings_info(
    config_path: Path,
    hardware_name: str,
    corpus_name: str,
    objective: str,
) -> str:
    """Format settings information.
    
    Args:
        config_path: Path to config file
        hardware_name: Hardware name
        corpus_name: Corpus name
        objective: Objective function string
        
    Returns:
        Formatted settings string
    """
    return "\n".join([
        f"loaded settings from {config_path}:",
        f"hardware = '{hardware_name}'",
        f"corpus = '{corpus_name}'",
        f"objective = '{objective}'",
        ""
    ])


def format_intro(
    settings_str: str,
    width: int = 80,
) -> str:
    """Format the intro message shown when starting the REPL.
    
    Args:
        settings_str: Formatted settings string
        width: Maximum line width
        
    Returns:
        Formatted intro message
    """
    return "\n".join([
        "",
        "",
        "--",
        "Jalo -- just another layout optimizer",
        "--",
        "",
        settings_str,
        "",
        "Tab does autocomplete, up arrow brings up previous commands. Type `help` to list commands.",
        ""
    ])


def format_metrics_table(
    metrics: Sequence[Any],
    break_before_metrics: Optional[set[str]] = None,
) -> str:
    """Format a table of metrics and their descriptions.
    
    Args:
        metrics: List of metric objects with .name and .description attributes
        break_before_metrics: Optional set of metric names that should have a blank row before them
        
    Returns:
        Formatted metrics table
    """
    break_before_metrics = break_before_metrics or set()
    header = ['Metric', 'Description']
    rows = []
    
    for metric in metrics:
        if metric.name in break_before_metrics:
            rows.append([None] * 2)
        rows.append([metric.name, metric.description])
    
    return format_table(rows, headers=header, tablefmt="simple")


def format_help_summary(
    headers: list[str],
    sections: dict[str, list[tuple[str, str]]],
    width: int = 70,
) -> str:
    """Format the general help summary showing all commands grouped by category.    """
    lines = [
        "",
        "Type `help <command>` to get help on a specific command.",
        ""
    ]
    
    for section_name in headers:
        if section_name not in sections or not sections[section_name]:
            continue
        lines.append(f"\n{section_name}:")
        lines.append(textwrap.indent(
            format_table(sections[section_name], tablefmt="plain", maxcolwidths=[width]),  # pyright: ignore[reportArgumentType]
            "  "
        ))
        lines.append("")
    
    return "\n".join(lines)


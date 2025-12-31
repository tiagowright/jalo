# Jalo

Jalo is just another keyboard layout optimizer.

## Installation

```bash
pip install .
```

## Usage

```bash
jalo
```

## Analysis

Jalo provides comprehensive analysis tools to understand keyboard layouts from multiple perspectives. The `analyze` command examines a single layout, displaying its structure, hardware configuration, and performance across all metrics, giving you a complete picture of how a layout performs. When you need to compare multiple layouts, `compare` presents them side by side, making it easy to see which layout excels in specific areas like finger effort, hand alternation, or bigram frequency. To understand what's driving a layout's overall score, `contributions` breaks down the score into metric-by-metric contributions, revealing which aspects are helping or hurting the layout's performance. You can explore available metrics with `metrics`, view or update the text corpus that determines character frequencies with `corpus`, and simply visualize a layout's structure with `show`. These tools work together to give you deep insights into keyboard layout performance, helping you understand not just how well a layout scores, but why it scores that way and what specific characteristics contribute to its effectiveness.

## Optimization

Jalo's optimization capabilities help you discover and refine keyboard layouts through multiple complementary approaches. The `generate` command creates entirely new layouts from scratch, starting from random seeds and iteratively improving them to find layouts with the lowest scores based on your objective function. This is perfect for exploring the solution space broadly and discovering novel layouts that might outperform existing ones. Once you have promising candidates, `improve` refines them further by making smaller, more conservative changes—producing layouts that are similar to the original but potentially better scoring. For final fine-tuning, `polish` identifies small numbers of strategic swaps that can improve a layout's score, perfect for those last few percentage points of optimization. Throughout this process, you can customize the `objective` function to define what makes a layout good—whether that's minimizing finger effort, maximizing hand alternation, or balancing multiple competing concerns. The `hardware` command lets you specify different keyboard physical configurations, and `pin` allows you to lock certain characters in place, ensuring that your optimization respects constraints like keeping vowels on the home row or maintaining specific ergonomic preferences.

## Editing

Jalo offers a suite of editing commands for manual refinement and experimentation with keyboard layouts. The `swap` command lets you make hand-crafted adjustments by exchanging positions between characters, perfect for fine-tuning layouts based on personal preference or addressing specific issues you've identified through analysis. For more dramatic transformations, `mirror` horizontally flips a layout, swapping the left and right hands entirely, while `invert` vertically mirrors the top and bottom rows, either for both hands or just one side. These transformations can reveal interesting variations or help adapt layouts for different typing styles. The `list` command helps you manage multiple layouts in memory, showing available layouts and their scores so you can track your experiments. When you're satisfied with a layout, `save` writes it to disk with a custom name, making it available for future sessions and allowing you to build a library of optimized layouts. Together, these editing tools give you complete control over layout manipulation, whether you're making subtle tweaks or exploring radical redesigns.

## Commands

Jalo includes essential utility commands that support your workflow and help you navigate the system. The `help` command provides detailed documentation for any command—simply type `help <command>` to get comprehensive usage information, examples, and argument descriptions. This built-in help system ensures you always have access to documentation without leaving the application. The `reload` command refreshes your configuration from the `config.toml` file, updating corpus settings, hardware defaults, and objective functions while preserving any layouts you've generated in memory. This is particularly useful when you want to experiment with different scoring configurations or switch between different text corpora without restarting the application. Finally, `exit` and `quit` both provide a clean way to close Jalo, ensuring your work is properly saved and the session ends gracefully. These utility commands round out Jalo's functionality, making it a complete, self-contained tool for keyboard layout optimization and analysis.
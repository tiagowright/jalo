# Jalo

Jalo is just another layout optimizer. Jalo can analyze keyboards with any number of keys and physical configurations, including keys on thumbs, any number of pinky or central columns, extra rows, keys on layers, and more. Optimization features can generate a wide range of layouts, improve existing layouts, and fine tune them by suggesting the most promising swaps. Under the hood, Jalo combines simulated annealing, genetic algorithms, and different flavors of hill climbing depending on the job for the best outcomes. To score layouts, Jalo allows any (linear) combination of metrics with user-defined weights, and makes it simple and interactive to change them. Jalo also provides clear reporting on how each metric contributes to each layout's final score, so it's easy to understand how to improve both the scoring and the layouts. Most of the metrics in the Keyboard Layout doc are included, but the definitions of the metrics can be changed, and new metrics can be added. Jalo makes it easy to analyze layouts, identify what key combiniations are contributing to any metric, and compare layouts side-by-side. Editing layouts is simple, with commands to mirror and invert layouts, and swap pairs of keys.

Jalo works as a command line tool with an interactive mode (REPL), or by invoking it with commands or a script file. Start the interactive mode `./jalo.py` then type `help` to get a list of commands. Use tab to auto-complete command names and arguments, and up/down to cycle through command history.

## Examples

### Analyze

```
jalo> analyze enthium

enthium                      ortho_pinky_33
---------------------------  ----------------------------
|   z p d l x   = u o y q    |    0 1 2 3 3   6 6 7 8 9
| w s n t h k   - e a i c b  |  0 0 1 2 3 3   6 6 7 8 9 9
|   v f g m j   ; / . , '    |    0 1 2 3 3   6 6 7 8 9
|           r                |            4

metric                     enthium  top n-gram
                    ortho_pinky_33
----------------  ----------------  -------------------------------------------------
rep                          2.742  ll: 0.713, ee: 0.412, ss: 0.353, oo: 0.295
sfb                          0.795  ue: 0.128, y,: 0.104, oa: 0.074, nf: 0.059
sfs                          5.795  dt: 0.491, gt: 0.454, ue: 0.434, oa: 0.326
sft                          0.007  e-e: 0.002, e-u: 0.001, ue-: 0.001, ueu: 0.000
...
```

### Compare

```
jalo> compare qwerty graphite hdpm
metric              qwerty  Δ      graphite  Δ                hdpm  Δ
                      ansi             ansi         ortho_thumb_33
----------------  --------  ---  ----------  ---  ----------------  ---
rep                  2.733            2.741                  2.742
sfb                  6.604  +         0.996  -               0.828  -
sfs                 11.238  +         6.260  -               6.653  -
sft                  0.431  +         0.025  -               0.005  -
...
```

### Generate layouts

```
jalo> objective
Current function:
objective 100sfb + 6effort + 60pinky_ring + 60scissors_ortho + 60sfs + 20finger_0 + 20finger_9 + 18finger_1 + 18finger_8 + 15finger_2 + 15finger_7 + 12finger_3 + 12finger_6

jalo> generate
generating 100 seeds.
Generating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:24<00:00,  4.15it/s]

layout list 1 > generate

layout 1.1 2931.348
x w l m k   q h o - ;
s c r t v   f n a e i
y g , d j   b p . ' u

layout 1.2 2933.606
; - o r j   k d . g v
i e a n x   f h t c s
u , ' l q   b p m w y
...
```

### Understand the scores

```
jalo> contributions 1.1 graphite sturdy
metric                  w      1.1  Δ          ΔS    graphite  Δ          ΔS    sturdy  Δ          ΔS
                              ansi                       ansi                     ansi
----------------  -------  -------  ---  --------  ----------  ---  --------  --------  ---  --------
rep                          2.735                      2.741                    2.732
sfb               100.000    0.980         98.043       0.996         99.557     0.870         87.021
sfs                60.000    6.123        367.408       6.260        375.571     5.996        359.783
...
score                                    2931.348                   3030.263                 2996.416
```

### Help

```
jalo> help

Type `help <command>` to get help on a specific command.


analysis:
  analyze        analyze a keyboard layout on all metrics
  compare        compares keyboard layouts side by side on every metric
  contributions  to understand the score, tabulates the contributions of each metric
  corpus         view or update the text corpus used to score layouts
  metrics        shows metric names and descriptions
  show           show a keyboard layout


optimization:
  generate   generates a wide variety of new layouts from scratch
  hardware   view or update the default keyboard hardware used to generate layouts
  improve    try to improve the score of a given layout (neighboring layouts)
  objective  view or update the objective function used to score layouts
  pin        pins characters to their current position
  polish     identifies small number of swaps that can improve the score of a given layout


editing:
  invert  inverts top and bottom rows of a keyboard layout (mirrors vertically)
  list    lists layouts in memory
  mirror  mirrors a keyboard layout horizontally
  save    save new layouts from memory to a new file
  swap    swaps two or more positions on the keyboard
...
```

## Installation

```bash
git clone https://github.com/tiagowright/jalo
cd jalo
pip install .
```

## Usage

```bash
./jalo.py
```


## Analysis

Jalo provides comprehensive analysis tools to understand keyboard layouts from multiple perspectives. The `analyze` command examines a single layout, displaying its structure, hardware configuration, and performance across all metrics, giving you a complete picture of how a layout performs. When you need to compare multiple layouts, `compare` presents them side by side, making it easy to see which layout excels in specific areas like finger effort, hand alternation, or bigram frequency. To understand what's driving a layout's overall score, `contributions` breaks down the score into metric-by-metric contributions, revealing which aspects are helping or hurting the layout's performance. You can explore available metrics with `metrics`, view or update the text corpus that determines character frequencies with `corpus`, and simply visualize a layout's structure with `show`. These tools work together to give you deep insights into keyboard layout performance, helping you understand not just how well a layout scores, but why it scores that way and what specific characteristics contribute to its effectiveness.

## Optimization

Jalo's optimization capabilities help you discover and refine keyboard layouts through multiple complementary approaches. The `generate` command creates entirely new layouts from scratch, starting from random seeds and iteratively improving them to find layouts with the lowest scores based on your objective function. This is perfect for exploring the solution space broadly and discovering novel layouts that might outperform existing ones. Once you have promising candidates, `improve` refines them further by making smaller, more conservative changes—producing layouts that are similar to the original but potentially better scoring. For final fine-tuning, `polish` identifies small numbers of strategic swaps that can improve a layout's score, perfect for those last few percentage points of optimization. Throughout this process, you can customize the `objective` function to define what makes a layout good—whether that's minimizing finger effort, maximizing hand alternation, or balancing multiple competing concerns. The `hardware` command lets you specify different keyboard physical configurations, and `pin` allows you to lock certain characters in place, ensuring that your optimization respects constraints like keeping vowels on the home row or maintaining specific ergonomic preferences.

## Editing

Jalo offers a suite of editing commands for manual refinement and experimentation with keyboard layouts. The `swap` command lets you make hand-crafted adjustments by exchanging positions between characters, perfect for fine-tuning layouts based on personal preference or addressing specific issues you've identified through analysis. For more dramatic transformations, `mirror` horizontally flips a layout, swapping the left and right hands entirely, while `invert` vertically mirrors the top and bottom rows, either for both hands or just one side. These transformations can reveal interesting variations or help adapt layouts for different typing styles. The `list` command helps you manage multiple layouts in memory, showing available layouts and their scores so you can track your experiments. When you're satisfied with a layout, `save` writes it to disk with a custom name, making it available for future sessions and allowing you to build a library of optimized layouts. Together, these editing tools give you complete control over layout manipulation, whether you're making subtle tweaks or exploring radical redesigns.

## Commands

Jalo includes essential utility commands that support your workflow and help you navigate the system. The `help` command provides detailed documentation for any command—simply type `help <command>` to get comprehensive usage information, examples, and argument descriptions. This built-in help system ensures you always have access to documentation without leaving the application. The `reload` command refreshes your configuration from the `config.toml` file, updating corpus settings, hardware defaults, and objective functions while preserving any layouts you've generated in memory. This is particularly useful when you want to experiment with different scoring configurations or switch between different text corpora without restarting the application. Finally, `exit` and `quit` both provide a clean way to close Jalo, ensuring your work is properly saved and the session ends gracefully. These utility commands round out Jalo's functionality, making it a complete, self-contained tool for keyboard layout optimization and analysis.
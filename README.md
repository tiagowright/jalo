# Jalo: just another layout optimizer

Jalo is an interactive keyboard layout analyzer and optmizer, with the following capabilities;

* *Flexible number of keys*: Jalo can analyze keyboards with any number of keys and physical configurations, including keys on thumbs, any number of pinky or central columns, extra rows, keys on layers, and more. 
* *Powerful optimization*: Optimization features can generate a wide range of layouts, improve existing layouts, and fine tune them by suggesting the most promising swaps. Under the hood, Jalo combines simulated annealing, genetic algorithms, and different flavors of hill climbing depending on the job.
* *Your own scoring function*: To score layouts, Jalo allows any (linear) combination of metrics with user-defined weights, and makes it simple and interactive to change them. Jalo also provides clear reporting on how each metric contributes to each layout's final score, so it's easy to understand how to improve both. 
* *Your metrics*: Most of the metrics in the [Keyboard layout doc](https://docs.google.com/document/d/1W0jhfqJI2ueJ2FNseR4YAFpNfsUM-_FlREHbpNGmC2o) are included, but the definitions of the metrics can be changed, and new metrics can be added. Jalo makes it easy to analyze layouts, identify what key combiniations are contributing to any metric, and compare layouts side-by-side. 
* *Editing*: Editing layouts is simple, with commands to mirror and invert layouts, and swap pairs of keys.

Jalo works as a command line tool with an interactive mode (REPL), or by invoking it with commands or a script file. Start the interactive mode `./jalo.py` then type `help` to get a list of commands. Use tab to auto-complete command names and arguments, and up/down to cycle through command history.

## Installation

To install, clone this repo, then use pip to install dependencies (see `pyproject.toml` if you want to inspect them).

```bash
git clone https://github.com/tiagowright/jalo
cd jalo
pip install .
```

## Usage

Once installed, you can use Jalo by invoking it from the command line:

```bash
./jalo.py
```

At the prompt `jalo> `, type `help` for an overview of all available commands, and `help <command>` for details on each command. Use arrows up/down to cycle through your command history. Tab is auto-complete, and it will complete commands as well as arguments, including completing layout names, metric names, etc.

The typical workflows are to `analyze` and `compare` layouts, or to `generate`, `improve`, `polish`, modify layouts (e.g. `swap`) for your needs. In the sections that follow, we'll cover these in more depth.


## Analyzing layouts

Jalo analyzes layouts and reports on 30+ common metrics, plus finger-level metrics. Use `analyze` to see how a specific layout fare against all metrics, along with the key sequences that are the most impactful on that metric. For example, `analyze enthium` will shows that `sfb` is 0.795% and that the worst offender is `ue` at 0.128% frequency in English.

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

You can also compare multiple layouts on all metrics with `compare`. For example, `compare graphite sturdy` will quickly show that graphite has more `alt_sfs` but lower redirects. 

```
jalo> compare graphite sturdy
metric              graphite  Δ      sturdy  Δ
                        ansi           ansi
----------------  ----------  ---  --------  ---
...
same_hand              6.510  -       9.493  +
alt                   37.295  +      31.755  -
alt_sfs                4.854  +       3.898  -
alt_total             42.149  +      35.654  -
...
redirect               2.037  -       3.338  +
redirect_sfs           0.549  -       1.569  +
redirect_bad           0.283          0.212
redirect_bad_sfs       0.114          0.110
redirect_total         2.983  -       5.229  +
...
```

You can also change the corpus being used to assess the metrics using the `corpus` command, for example, to assess layouts on different language, or to use your own corpus (see [Corpus section](#corpus)).


## Layouts

Jalo ships with a number of layouts, all available in `./layouts/` folder. You can also use tab to auto-complete layout names to get a list:

```
jalo> analyze 
asto       colemak    dvorak     enthium    graphite   hdneu      hdti       isrt       qwerty     sturdy    
asto_22    colemak_dh engram     gallium    hdgold     hdpm       inrolly    mtgap      semimak    workman   
```

Creating your own layout is easy. You can add a `.kb` file in the `./layouts/` folder using a text editor (use any of the existing files there as examples). 

You can also create layouts in Jalo, for example, using `generate` or by editing an existing layout with `swap`, `mirror`, `invert`, and so on. Layouts created in jalo are kept in memory and can be accessed by their number (use `list` command to see what is in memory). If you are happy with a layout, you can use `save` to create a new file in the `./layouts/` folder that can be referenced later, edited by hand for any additional changes, shared, etc.

```
jalo> invert hdpm

layout 2 3004.163 (30.686)  > invert hdpm
v w g m j   - . ' = / z
s n t h k   , a e i c q
f p d l x   ; u o y b  
        r            

jalo> save 2 hdpm_inverted
saved layout to layouts/hdpm_inverted.kb
```

Layout files can optionally name the hardware to be used for the layout. Many layouts are created using a 3 x 10 grid of 30 keys that are compatible with `ansi` or `ortho` hardware. When a layout is needed, jalo tries to load it into the default hardware, typically `ansi` or `ortho`. You can change the default with the `hardware` command, or by editing the `config.toml` file to change it for all future sessions.

But you may be interested in adding thumb letter keys, more keys to the pinky finger, removing keys, adding layers, etc. Layouts that are not compatible with the 3 x 10 grid can name the hardware setup to be used for that layout by adding a comment with a hint, for example, `# use: ortho_thumb` will tell jalo to use `ortho_thumb` hardware for this layout (see `hdpm.kb` for an example).

See [Keebs](#keebs) section for more information about customizing the hardware, and [Defaults](#defaults) to change the default hardware.

## Keebs

Jalo supports a variety of keyboard hardware with different numbers and arrangements of keys, available in the `./keebs/` folder, including:
* `ansi`: the standard 30 key, row staggered, setup common to most laptops, and the most widely used for layout designs
* `ortho`: the analogous 30 key orthonormal setup (no stagger), that is compatible with `ansi` so that any layout built for one can be easily transposed on the other
* `ansi_angle`: this is the standard `ansi` setup with an "angle mod", where the typist uses a different set of fingers for the bottom row on the left hand
* `ortho_thumb`: the `ortho` setup with an additional key on the left thumb that can assigned a letter/character, for layouts like Hands Down series

Jalo shows layouts and the hardware they are loaded on side by side. The hardware shows the fingers that are assigned to each position, from the left pinky at 0 to the right pinky at 9.

```
jalo> show qwerty

qwerty                   ansi
-----------------------  --------------------------
| q w e r t   y u i o p  |  0 1 2 3 3   6 6 7 8 9
| a s d f g   h j k l ;  |   0 1 2 3 3   6 6 7 8 9
| z x c v b   n m , . /  |    0 1 2 3 3   6 6 7 8 9

jalo> hardware ortho
Updated hardware to: ortho
jalo> show graphite

graphite                 ortho
-----------------------  ------------------------
| b l d w z   ' f o u j  |  0 1 2 3 3   6 6 7 8 9
| n r t s g   y h a e i  |  0 1 2 3 3   6 6 7 8 9
| q x m c v   k p . - ,  |  0 1 2 3 3   6 6 7 8 9

jalo> show hdpm

hdpm                       ortho_thumb_33
-------------------------  --------------------------
| f p d l x   ; u o y b z  |  0 1 2 3 3   6 6 7 8 9 9
| s n t h k   , a e i c q  |  0 1 2 3 3   6 6 7 8 9 9
| v w g m j   - . ' = /    |  0 1 2 3 3   6 6 7 8 9
|         r                |          4

jalo> show inrolly

inrolly                  ansi_angle
-----------------------  --------------------------
| y o u q x   k d l w ,  |  0 1 2 3 3   6 6 7 8 9
| i a e n j   v h t s c  |   0 1 2 3 3   6 6 7 8 9
| " - r b z   f p m g .  |    1 2 3 3 3   6 6 7 8 9
```

To create your own hardware setup, add a file to the `./keebs` directory that exports a `KEYBOARD` attribute. The exported object needs to be an instance of `hardware.KeyboardHardware`. The simplest way to build a new layout is to modify the ansi/ortho setup by adding or modifying keys (e.g. see `ortho_pinky_33`). If you must / want to start from scratch, use `ansi` or `cr8` as examples for the approach. Once the file is created, you can refer to the keyboard by the file's name in Jalo.


## Metrics

The metrics were carefully created to match as closely as possible the definitions in the [Keyboard layout doc](https://docs.google.com/document/d/1W0jhfqJI2ueJ2FNseR4YAFpNfsUM-_FlREHbpNGmC2o) and to work with a wide range of different kayboard hardware definitions. Use the `metric` command to review the metrics and their descriptions.

You can also inspect, update, or create your own metrics. All metrics are defined in `metrics.py`. Metrics are defined as a method that takes either 1, 2, or 3 key positions, and returns a floating point value. You can inspect the definitions and modify the functions to better align with your preferences if needed. For instance, `scissors` are especially difficult to define, and you may have different preferences for what should be included or not.

To create your own metric, add a new function to `metrics.py`, then add your new metric to the `METRICS` array at the end of the file. Copy the patterns of an existing metric that is of the same type. If you define a new trigram based metric (three position inputs), it may be worth thinking about efficiency in your code, since these are computed for every combination of 3 positions on the keyboard you are analyzing (`O(N^3)`). After defining your metric, use `analyze` to check on the most common key combinations to inspect the quality of the definition.

```
jalo> metrics
Metric            Description
----------------  ------------------------------------------------------------------------
rep               single finger repetition
sfb               single finger bigram
sfs               single finger skipgram
...
```

## Scoring layouts

The "magic" in optimizing layouts using most computer optimization tool is figuring out how to score them. The workflow often involves coming up with a way to score layouts (such as setting a bunch of weights), running the optimization, then tweaking the weights again to improve the outputs. Jalo makes it easier to understand how the current score is calculate and what drives it, and then simple to update the scoring function.

In Jalo, the score is an amount of *heat*, so that lower scores are always considered better. This means that "bad" things should have a positive weight, and "good" things a negative weight.

Jalo allows any linear combination of any of it's metrics to create a score. Typically, this is a combination of positive weights (indicating increased heat) for metrics such as `sfb`, `sfs`, `scissors`, `redirect`, and negative weights (indicating reduced heat) for metrics such as `in_roll`, `home`, `alt`. This combination of weights into a single score is called an "objective".

Jalo ships with some objectives that are available in `./objectives`. Use the `objective` command to name one of these as your objective function (e.g., `objective default` for the "default" function), or update `config.toml` to change the default for all future sessions as well.

Jalo makes it easy to create your own objectives. Use the `objective` command to view the current objective, and use it to make updates to the weights, add / remove metrics, etc directly in the interative shell.

```
jalo> objective
Current function:
objective 100sfb + 6heat + 60pinky_ring + 60scissors_ortho + 60sfs + 20finger_0 + 20finger_9 + 18finger_1 + 18finger_8 + 15finger_2 + 15finger_7 + 12finger_3 + 12finger_6

jalo> objective 100sfb + 60sfs + 60pinky_ring
Updated function to:
objective 100sfb + 60sfs + 60pinky_ring
```

If you have a good objective function, you can also save it, and use it in the future, or set it as the default. Create a new `.toml` file in `./objectives` with the name you want. The toml file must follow one of these two conventions:

* Have a key called `formula` with a string that contains the same format of formula that the `objective` command supports (e.g., `default.toml`)
* Have a `formula` table, where each key is the name of a metric, and the value is the weight assigned to that metric (e.g., `oxeylyzer.toml`)

Jalo makes it much easier to understand how the layouts you are building are getting scored, in detail. A tricky step in improving the objective can be just understanding why it is leading to some layouts that are not great, what weights are driving it there. Jalo provides `contributions` command, which breaks down how each weight and metric contributes to the score, comparing multiple layouts side-by-side. 

In this example, layout `1.1` is compared with `hdpm` and `graphite`, with the score weights shown in `w`, and the contributions to the score for each metric and layout in the `ΔS` columns, showing that `sfs` in this case is one of the drivers of the differences in score.

```
jalo> contributions 1.1 hdpm graphite
metric                  w               1.1  Δ          ΔS              hdpm  Δ          ΔS    graphite  Δ          ΔS
                             ortho_thumb_33                   ortho_thumb_33                       ansi
----------------  -------  ----------------  ---  --------  ----------------  ---  --------  ----------  ---  --------
rep                                   2.742                            2.742                      2.741
sfb               100.000             0.906         90.626             0.828         82.792       0.996         99.557
sfs                60.000             5.162  -     309.722             6.653  +     399.162       6.260        375.571
...
score                                             2657.097                         2973.477                   3030.263
```

## Generating layouts

TODO

## Improving and Polishing

TODO

## Corpus

When analyzing a layout, the corpus determines how often key sequences of a layout will be typed, and thus it is a critical driver of all the results. For example, the `en` corpus (the default English language corpus) specifies that the sequence of letters `ed` will be typed 0.99% of the time, and therefore, on a qwerty layout, where `e` and `d` are both on the left middle finger, it drives the `sfb` metric (single finger bigram) up by that amount.

Jalo corpuses are provided in the `./corpus` directory, including the most commonly used corpus for english language optimizations. To select a corpus, use the `corpus` command, or to avoid repeating the command, change the default corpus in `config.toml` ([see Defaults section](#defaults))

To provide your own corpus, create a new folder under `./corpus` named for the new corpus. The folder should contain 4 json files: `monograms.json`, `bigrams.json`, `trigrams.json`, `skipgrams.json`, that specify the frequency of each ngram in the corpus. The values in each file should add up to 1.0 (i.e., should be frequencies, not counts). Additional files can be added for more information about the corpus, and will be dutifully ignored.

Creating a great corpus can be tricky to get right. You want to the corpus to be closely representative of what you will be typing, and large enough to provide robust statistics within a tenth of a percent or better. The corpus also needs to be cleaned, for example, by dealing with capitalization, accented characters, special characters that are shifted, and so on. I would recommend using an existing corpus, either provided here, or from one of the other layout optimization projects listed in references, if possible.

## Defaults

Default values are loaded from `config.toml` and define the default hardware, corpus, and objective function that will be used. These can later be changed with `hardware`, `corpus`, and `objective` commands, but it's worth editing the file to put in your preferred values and avoid having to set them each time.

The config also allows you to set the `oxeylyzer_mode` flag, which will force Jalo to use metric definitions that match exactly with the definitions in [Oxeylyzer](https://github.com/o-x-e-y/oxeylyzer/), where ever possible. Where the Jalo definition is different, it is only because I believe it is a better reflection of what is described in the [Keyboard layout doc](https://docs.google.com/document/d/1W0jhfqJI2ueJ2FNseR4YAFpNfsUM-_FlREHbpNGmC2o), but you may have a different opinion (see Metrics section as well).

## Examples

### Analyze a layout

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
objective 100sfb + 6heat + 60pinky_ring + 60scissors_ortho + 60sfs + 20finger_0 + 20finger_9 + 18finger_1 + 18finger_8 + 15finger_2 + 15finger_7 + 12finger_3 + 12finger_6

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

## Optimization

Jalo's optimization capabilities help you discover and refine keyboard layouts through multiple complementary approaches. The `generate` command creates entirely new layouts from scratch, starting from random seeds and iteratively improving them to find layouts with the lowest scores based on your objective function. This is perfect for exploring the solution space broadly and discovering novel layouts that might outperform existing ones. Once you have promising candidates, `improve` refines them further by making smaller, more conservative changes—producing layouts that are similar to the original but potentially better scoring. For final fine-tuning, `polish` identifies small numbers of strategic swaps that can improve a layout's score, perfect for those last few percentage points of optimization. Throughout this process, you can customize the `objective` function to define what makes a layout good—whether that's minimizing finger heat, maximizing hand alternation, or balancing multiple competing concerns. The `hardware` command lets you specify different keyboard physical configurations, and `pin` allows you to lock certain characters in place, ensuring that your optimization respects constraints like keeping vowels on the home row or maintaining specific ergonomic preferences.

## Editing

Jalo offers a suite of editing commands for manual refinement and experimentation with keyboard layouts. The `swap` command lets you make hand-crafted adjustments by exchanging positions between characters, perfect for fine-tuning layouts based on personal preference or addressing specific issues you've identified through analysis. For more dramatic transformations, `mirror` horizontally flips a layout, swapping the left and right hands entirely, while `invert` vertically mirrors the top and bottom rows, either for both hands or just one side. These transformations can reveal interesting variations or help adapt layouts for different typing styles. The `list` command helps you manage multiple layouts in memory, showing available layouts and their scores so you can track your experiments. When you're satisfied with a layout, `save` writes it to disk with a custom name, making it available for future sessions and allowing you to build a library of optimized layouts. Together, these editing tools give you complete control over layout manipulation, whether you're making subtle tweaks or exploring radical redesigns.

## Commands

Jalo includes essential utility commands that support your workflow and help you navigate the system. The `help` command provides detailed documentation for any command—simply type `help <command>` to get comprehensive usage information, examples, and argument descriptions. This built-in help system ensures you always have access to documentation without leaving the application. The `reload` command refreshes your configuration from the `config.toml` file, updating corpus settings, hardware defaults, and objective functions while preserving any layouts you've generated in memory. This is particularly useful when you want to experiment with different scoring configurations or switch between different text corpora without restarting the application. Finally, `exit` and `quit` both provide a clean way to close Jalo, ensuring your work is properly saved and the session ends gracefully. These utility commands round out Jalo's functionality, making it a complete, self-contained tool for keyboard layout optimization and analysis.

## References

Thank you to the Keyboard layout communities for your help and inspiration over the years. 

The following projects speficically were fundamental inspiration and sources of ideas for Jalo:
* [Keyboard layout doc](https://docs.google.com/document/d/1W0jhfqJI2ueJ2FNseR4YAFpNfsUM-_FlREHbpNGmC2o)
* [carpalx](https://mk.bcgsc.ca/carpalx/)
* [oxeylyzer](https://github.com/o-x-e-y/oxeylyzer/)
* [keygen](https://github.com/xsznix/keygen)
* [keymui](https://github.com/semilin/keymui)
* [cmini](https://github.com/Apsu/cmini)
* [colemak](https://colemak.com/Design)

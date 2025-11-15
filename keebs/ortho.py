r'''
Standard Ortho hardware, with 30 keys in 3 rows of 10 columns.

This is very similar to the ANSI standard hardware, but there is no row stagger,
which impacts primarily distance metrics (e.g., `dist`, `sfb_dist`).

Note that an often overlooked difference is `scissors` (FSB) metrics. Typical definitions
are based on ANSI hardware with row stagger, which means for example that qwerty `in` is not considered
scissors, but `eb` often is. In an ortho hardware, these are mirrored images of each other,
and if you consider `eb` as scissors, then `in` should be too. My recommendation is to use a different 
definition of scissors that is fully symmetric, such as the included `scissors_ortho`.

ortho:
 0 1 2 3 3   6 6 7 8 9
 0 1 2 3 3   6 6 7 8 9
 0 1 2 3 3   6 6 7 8 9
'''


from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware('ortho', stagger_at_row={})

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
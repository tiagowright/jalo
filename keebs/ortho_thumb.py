r'''
Standard Ortho hardware, with 30 keys in 3 rows of 10 columns, plus a thumb key.

Some layouts take advantage of the thumb key, assigning a high frequency letter to it,
which reduces SFBs and improves typing speed, but obviously requires typing a letter on thumb.
Example: `./layouts/hdtitanium.kb`
'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware('ortho', stagger_at_row={}, additional_positions=[
    Position(row=3, col=4, x=3, y=4, finger=Finger.LT, is_home=True, effort=1.0),
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True))
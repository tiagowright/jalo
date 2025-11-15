r'''
Standard Ortho hardware, with 30 keys in 3 rows of 10 columns, plus a thumb key.

Some layouts take advantage of the thumb key, assigning a high frequency letter to it,
which reduces SFBs and improves typing speed, but obviously requires typing a letter on thumb.
Example: `./layouts/hdtitanium.kb`

ortho_thumb:
 0 1 2 3 3   6 6 7 8 9
 0 1 2 3 3   6 6 7 8 9
 0 1 2 3 3   6 6 7 8 9
         4    
'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware('ortho_thumb', stagger_at_row={}, additional_positions=[
    Position(row=3, col=4, x=4, y=3, finger=Finger.LT, is_home=True, effort=1.0),
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True))
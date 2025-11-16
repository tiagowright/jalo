r'''
Standard ANSI hardware, with 2 additional keys on the right pinky (qwerty `{` and `;`)

Some layouts take advantage of the extra keys to move less used letters out, and put
more common punctuation into the center, e.g. `./layouts/hdneu.kb`

ansi_32:
 0 1 2 3 3   6 6 7 8 9 9
  0 1 2 3 3   6 6 7 8 9 9
   0 1 2 3 3   6 6 7 8 9  

'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware, stagger_at_row

KEYBOARD = standard_hardware(additional_positions=[
    Position(row=0, col=10, x=10, y=0, finger=Finger.RP, is_home=False, effort=3.2),
    Position(row=1, col=10, x=10 + stagger_at_row[1], y=1, finger=Finger.RP, is_home=False, effort=2.8)
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True))
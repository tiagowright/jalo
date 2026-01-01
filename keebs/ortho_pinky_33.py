r'''
Ortho hardware, with an extra keys on each pinky plus a left thumb key.

See: `ansi_32.py` for the standard ANSI hardware with the extra pinky keys.
See: `ortho_thumb_33.py` with two right pinky keys and thumb key.

   0 1 2 3 3   6 6 7 8 9
 0 0 1 2 3 3   6 6 7 8 9 9
   0 1 2 3 3   6 6 7 8 9
           4    
'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware(stagger_at_row={}, additional_positions=[
    Position(row=1, col=-1, x=-1, y=0, finger=Finger.LP, is_home=False, effort=2.8),
    Position(row=1, col=10, x=10, y=1, finger=Finger.RP, is_home=False, effort=2.8),
    Position(row=3, col=4, x=4, y=3, finger=Finger.LT, is_home=True, effort=1.0)
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True))
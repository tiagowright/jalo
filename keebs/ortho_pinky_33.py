r'''
Ortho hardware, with an extra key on each pinky plus a right thumb key.

See: `ansi_32.py` for the standard ANSI hardware with the extra pinky keys.
See: `ortho_thumb_33.py` with two right pinky keys and thumb key.

   0 1 2 3 3   6 6 7 8 9
 0 0 1 2 3 3   6 6 7 8 9 9
   0 1 2 3 3   6 6 7 8 9
               5    
'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware(stagger_at_row={}, additional_positions=[
    Position(row=1, col=-1, x=-1, y=1, finger=Finger.LP, is_home=False, heat=2.8),
    Position(row=1, col=10, x=10, y=1, finger=Finger.RP, is_home=False, heat=2.8),
    Position(row=3, col= 5, x=5,  y=3, finger=Finger.RT, is_home=True,  heat=1.0)
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True))

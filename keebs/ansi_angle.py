r'''
Standard ANSI hardware, with 30 keys in 3 rows of 10 columns, where the user is applying
an `angle mod`: the bottom row of the left hand is typed with the middle finger, ring 
finger, and index finger, respectively. 

In qwerty, z is typed with the middle finger, x with the ring finger, and c with the index finger.

Some layouts are specifically designed to take advantage of angle mod, where the left pinky
has only 2 positions, and the left index has 7 instead of 6.

Example: `./layouts/inrolly.kb``

ansi_angle:
 0 1 2 3 3   6 6 7 8 9
  0 1 2 3 3   6 6 7 8 9
   1 2 3 3 3   6 6 7 8 9
'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

# this is a standard ansi keyboard with an angle mod:
# the bottom row of the left hand is typed z by the middle finger, 
# x by the ring finger, and c by the index finger.
finger_at_row_col = {
    (2,0): Finger.LR,
    (2,1): Finger.LM,
    (2,2): Finger.LI,
}

KEYBOARD = standard_hardware('ansi_angle', finger_at_row_col=finger_at_row_col)

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
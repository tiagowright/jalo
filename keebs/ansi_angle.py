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
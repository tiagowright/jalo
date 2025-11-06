from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import standard_hardware

KEYBOARD = standard_hardware('ortho', stagger_at_row={})

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
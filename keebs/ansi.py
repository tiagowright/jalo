r'''
Standard ANSI hardware, with 30 keys in 3 rows of 10 columns, with a row stagger.
This is the typical setup for keyboard layout optimization.

This file also defines a function that can be used to create similar layouts
with different stagger patterns, finger assignments, home keys, heat maps, 
and additional positions.

ansi:
 0 1 2 3 3   6 6 7 8 9
  0 1 2 3 3   6 6 7 8 9
   0 1 2 3 3   6 6 7 8 9
'''

import sys
import inspect
import re

from hardware import Finger, KeyboardHardware, Position

finger_at_col = {
    0: Finger.LP,
    1: Finger.LR,
    2: Finger.LM,
    3: Finger.LI,
    4: Finger.LI,
    5: Finger.RI,
    6: Finger.RI,
    7: Finger.RM,
    8: Finger.RR,
    9: Finger.RP
}

stagger_at_row = {
    0: 0,
    1: 0.25,
    2: 0.75
}

is_home = {
    (1,0): True,
    (1,1): True,
    (1,2): True,
    (1,3): True,
    (1,6): True,
    (1,7): True,
    (1,8): True,
    (1,9): True
}

# heat map obtained from a linear regression on the speed of a key press using a key logger
# note that this does not account for the distances and the effects of the horizontal stagger
# which should be incorporated into your optimization using the distance metrics
heat_map = [
    [3.0, 1.6, 1.4, 1.2, 2.4,  2.4, 1.2, 1.4, 1.6, 3.0],
    [1.6, 1.4, 1.3, 1.0, 1.8,  1.8, 1.0, 1.3, 1.4, 1.6],
    [2.2, 1.8, 1.6, 1.3, 2.4,  2.4, 1.3, 1.6, 1.8, 2.2],
]

def standard_hardware(
    name: str | None = None, 
    stagger_at_row = stagger_at_row, 
    stagger_at_col = {}, 
    finger_at_col = finger_at_col, 
    finger_at_row_col = {},
    is_home = is_home,
    heat_map = heat_map,
    additional_positions = []
):
    '''Creates KeyboardHardware with a standard 30 key positions arranged in 3 rows of 10 columns
    
    This is the typical setup for keyboard layout optimization.

    Parameters
    ----------
    name : str
        The name of the keyboard hardware (defaults to the filename in ./keebs/)
    stagger_at_row : dict {int: float}
        The amount of x-axis stagger at each row, in U units (1U = 19.05mm).
        Defaults to the ANSI standard stagger.
    stagger_at_col : dict {int: float}
        The amount of y-axis stagger at each col, in U units (1U = 19.05mm).
        Defaults to no stagger.
    finger_at_col : dict {int: Finger}
        The finger at each column, if finger_at_row_col is not provided.
        Defaults to the ANSI standard finger assignment.
    finger_at_row_col : dict {(int, int): Finger}
        The finger assigned at each row and column. If not provided, finger_at_col is used.
    is_home : dict {(int, int): bool}
        Whether the key at the given row and column is a home key for that finger.
        Defaults to the ANSI standard home keys.
    heat_map : list[list[float]]
        The heat map for each key, used to calculate the `heat` metric.
        A rasonable default is provided, based on a linear regression of a user's key press speed.
    additional_positions : list[Position]
        Additional Positions to add to the keyboard hardware, for simple modifications,
        such as adding a thumb key or extra pinky keys (e.g. in Hands Down layouts)
    '''

    if not name:
        name = str(inspect.currentframe().f_back.f_globals.get('__name__', ''))  # pyright: ignore[reportOptionalMemberAccess]
        # remove all prefixes and suffixes from the name
        name = re.sub(r'^.*\.', '', name).strip()


    return KeyboardHardware(name=str(name), positions=[
        Position(
            row=row, 
            col=col, 
            x=col + stagger_at_row.get(row,0), 
            y=row + stagger_at_col.get(col,0), 
            finger=finger_at_row_col.get((row,col), finger_at_col[col]), 
            is_home=is_home.get((row,col), False),
            heat=heat_map[row][col]
        )
        for row in range(3)
        for col in range(10)
    ] + additional_positions)

# export the standard ansi keyboard
# KEYBOARD = standard_hardware('ansi')
KEYBOARD = standard_hardware()

# run as a module for imports to work
# python3 -m keebs.ansi
if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))

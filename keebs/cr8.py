'''
cr8 is a 22 key keeb, with 21 positions in the first layer and 9 positions in a second layer, 
so it can be optimized for the standard 30 keys.

the second layer is also designed to have the arrows in the right hand, so the positions of the
8 keys are not all at home.

you probably don't want to use this directly, but might find it useful as a reference for how
to create a custom keeb with layers

    1  2  3          66 7  88   
 0  11 22 3          6  7  8  99
 0  1  22            66 77 88 9 

'''

from hardware import Finger, KeyboardHardware, Position
from keebs.ansi import effort_map

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

cols_at_row = [
       [1, 2, 3, 6, 7, 8],
    [0, 1, 2, 3, 6, 7, 8, 9],
    [0, 1, 2,    6, 7, 8, 9]
]

second_layer_positions = [
    [6, 8],
    [1, 2, 9],
    [2, 6, 7, 8]
]

# additional effort needed to access the second layer (e.g., hold/tap the layer key)
# this will be added to the standard effort map for that row and column
second_layer_additional_effort = 1.0

KEYBOARD = KeyboardHardware(
    name='cr8', 
    positions=[
        Position(row=row, col=col, x=col, y=row, 
            finger=finger_at_col[col], 
            is_home = is_home.get((row, col), False), 
            effort=effort_map[row][col]
        )
        for row, cols in enumerate(cols_at_row)
        for col in cols
    ] + [
        Position(row=row, col=col, x=col, y=row, 
            finger=finger_at_col[col], 
            is_home = is_home.get((row, col), False), 
            layer = 1, 
            effort=effort_map[row][col] + second_layer_additional_effort
        )
        for row, cols in enumerate(second_layer_positions)
        for col in cols
    ]
)

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
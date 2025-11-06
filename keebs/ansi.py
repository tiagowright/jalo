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

def standard_hardware(name, stagger_at_row = stagger_at_row, stagger_at_col = {}):
    '''Creates KeyboardHardware with a standard 30 key positions arranged in 3 rows of 10 columns
    This is the typical setup for keyboard layout optimization.
    '''
    return KeyboardHardware(name=name, positions=[
        Position(
            row=row, 
            col=col, 
            x=col + stagger_at_row.get(row,0), 
            y=row + stagger_at_col.get(col,0), 
            finger=finger_at_col[col], 
            is_home=is_home.get((row,col), False)
        )
        for row in range(3)
        for col in range(10)
    ])

# export the standard ansi keyboard
KEYBOARD = standard_hardware('ansi')

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
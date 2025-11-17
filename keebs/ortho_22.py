'''
ortho_22:
   1 2 3       6 7 8  
 0 1 2 3       6 7 8 9
 0 1 2 3       6 7 8 9
'''


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

cols_at_row = [
       [1, 2, 3, 6, 7, 8],
    [0, 1, 2, 3, 6, 7, 8, 9],
    [0, 1, 2, 3, 6, 7, 8, 9]
]

KEYBOARD = KeyboardHardware(name='ortho_22', positions=[
    Position(row=row, col=col, x=col, y=row, finger=finger_at_col[col])
    for row, cols in enumerate(cols_at_row)
    for col in cols
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
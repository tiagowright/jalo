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

KEYBOARD = KeyboardHardware(name='ansi', positions=[
    Position(row=row, col=col, x=col + stagger_at_row[row], y=row, finger=finger_at_col[col])
    for row in range(3)
    for col in range(10)
])

if __name__ == "__main__":
    print(KEYBOARD.str(show_finger_numbers=True, show_stagger=True))
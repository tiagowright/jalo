import numpy as np
from collections import defaultdict
from typing import List
import os
from hardware import Finger, KeyboardHardware, Position
from dataclasses import dataclass

DEFAULT_HARDWARE = 'ortho'

@dataclass(frozen=True)
class LayoutKey:
    """
    Represent a key on a keyboard layout.
    """
    char: str
    row: int
    col: int
    x: float
    y: float
    finger: Finger
    position: Position

    @classmethod
    def from_position(cls, position: Position, char: str) -> 'LayoutKey':
        return cls(char, position.row, position.col, position.x, position.y, position.finger, position)


def _name_or_hardware(name_or_hardware: str | KeyboardHardware) -> KeyboardHardware:
    """Convert a hardware name string or KeyboardHardware instance to KeyboardHardware."""
    if isinstance(name_or_hardware, str):
        return KeyboardHardware.from_name(name_or_hardware)
    return name_or_hardware


class KeyboardLayout:
    def __init__(self, keys: List[LayoutKey], hardware: KeyboardHardware, name: str):
        self.keys = keys
        self.hardware = hardware
        self.name = name
        
        # validate that keys match hardware: every key points to a position and every position is pointed to by a key   
        position_set = set(key.position for key in keys)
        if len(position_set) != len(keys):
            raise ValueError("Keys must point to unique positions")
        if len(position_set) != len(hardware.positions):
            raise ValueError("Keys must point to all positions in hardware")
        
        self.char_to_key = {key.char: key for key in keys}
        self.key_at_position = {key.position: key for key in keys}
        self.char_at_position = {key.position: key.char for key in keys}

        self.grid = defaultdict(lambda: defaultdict(list[LayoutKey])) # row -> col -> keys
        for key in keys:
            self.grid[key.row][key.col].append(key) 

    def __repr__(self) -> str:
        return f"KeyboardLayout(keys={self.keys!r}, hardware='{self.hardware.name}', name='{self.name}')"
    
    def __str__(self) -> str:
        COL_SEP = ' '
        HAND_SEP = ' '
        BLANK_KEY = ' '
        
        # identify the longest char in the layout
        longest_char = max(len(key.char) for key in self.keys)
        
        # format the layout with the longest char width
        format_str = f"{{:<{longest_char}}}"

        # get the range of rows and cols
        min_row = min(self.grid.keys())
        max_row = max(self.grid.keys())
        min_col = min(col for row in self.grid.keys() for col in self.grid[row].keys())
        max_col = max(col for row in self.grid.keys() for col in self.grid[row].keys())

        text_grid_lines = []
        for row in range(min_row, max_row + 1):
            row_str = []
            prev_hand = None
            hand = None
            for col in range(min_col, max_col + 1):
                if row in self.grid and col in self.grid[row]:
                    hand = self.grid[row][col][0].finger.hand
                    if hand is not None and prev_hand is not None and hand != prev_hand:
                        row_str.append(HAND_SEP)
                    prev_hand = hand

                    row_str.append(format_str.format(''.join(key.char for key in self.grid[row][col])))
                else:
                    row_str.append(format_str.format(BLANK_KEY))

            text_grid_lines.append(COL_SEP.join(row_str))

        return '\n'.join(text_grid_lines)


    @classmethod
    def from_name(cls, name: str, hardware: str | KeyboardHardware = DEFAULT_HARDWARE) -> 'KeyboardLayout':
        """Load a keyboard layout by name from ``layouts/``."""
        text_file_path = os.path.join('layouts', f'{name}.kb')
        with open(text_file_path, 'r') as file:
            text_grid = file.read()
        return cls.from_text(text_grid, hardware, name=name)

    @classmethod
    def from_text(cls, text_grid: str, hardware: str | KeyboardHardware = DEFAULT_HARDWARE, name: str = '') -> 'KeyboardLayout':
        """Create a keyboard layout from a text grid."""
        hardware = _name_or_hardware(hardware)
        
        occupied_positions = set()
        keys = []
        hardware_rows = iter(sorted(hardware.grid.keys()))
        text_grid_lines = text_grid.split('\n')

        for line in text_grid_lines:
            if not line.strip() or line.strip().startswith('#'):
                continue

            try:
                row = next(hardware_rows)
            except StopIteration:
                raise ValueError(f"Too many rows in text grid. Expected {len(hardware.grid)}, got {len(text_grid_lines)} lines")

            hardware_cols = iter(sorted(hardware.grid[row].keys()))

            for char in line.split():
                try:
                    col = next(hardware_cols)
                except StopIteration:
                    raise ValueError(f"Too many columns in text grid. Expected {len(hardware.grid[row])}, got {len(line.split())}")

                for position in hardware.grid[row][col]:
                    if position in occupied_positions:
                        continue
                    occupied_positions.add(position)
                    break
                else:
                    raise ValueError(f"Cannot find available positions ({row}, {col}) for character {char}. Occupied positions: {hardware.grid[row][col]!r}")
            
                keys.append(LayoutKey(char, row, col, position.x, position.y, position.finger, position))
            
            try:
                hardware_cols = next(hardware_cols)
            except StopIteration:
                pass
            else:
                raise ValueError(f"Too few columns in text grid. Expected {len(hardware.grid[row])}, got {len(line.split())}")

        try:
            hardware_rows = next(hardware_rows)
        except StopIteration:
            pass
        else:
            raise ValueError(f"Too few rows in text grid. Expected {len(hardware.grid)}, got {len(text_grid_lines)} lines")
        
        if not name:
            name = ''.join(key.char for key in keys[:6])

        return cls(keys, hardware, name)



if __name__ == "__main__":

    hw = KeyboardHardware.from_name('ansi_32')
    layout = KeyboardLayout.from_name('hdneu', hw)
    print(layout.name)
    print(str(layout))
    print()

    hw = KeyboardHardware.from_name('ansi_angle')
    layout = KeyboardLayout.from_name('inrolly', hw)
    print(layout.name)
    print(str(layout))
    print()

    hw = KeyboardHardware.from_name('ansi')
    layout = KeyboardLayout.from_name('qwerty', hw)
    print(layout.name)
    print(str(layout))
    print()

    hw = KeyboardHardware.from_name('ortho_thumb')
    layout = KeyboardLayout.from_name('hdgold', hw)
    print(layout.name)
    print(str(layout))
    print()

    hw = KeyboardHardware.from_name('ortho')
    layout = KeyboardLayout.from_name('qwerty', hw)
    print(layout.name)
    print(str(layout))
    print()

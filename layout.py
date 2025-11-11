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

        self.grid = defaultdict(lambda: defaultdict(list)) # row -> col -> keys
        for key in keys:
            self.grid[key.row][key.col].append(key) 

    def __repr__(self) -> str:
        return f"KeyboardLayout(keys={self.keys!r}, hardware='{self.hardware.name}', name='{self.name}')"
    
    def __str__(self) -> str:
        return '\n'.join(
            ' '.join(
                ''.join(
                    key.char 
                    for key in sorted(self.grid[row][col])
                ) 
                for col in sorted(self.grid[row].keys())
            )
            for row in sorted(self.grid.keys()) 
        )

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
        row = 0
        for line in text_grid.split('\n'):
            if not line.strip() or line.strip().startswith('#'):
                continue

            for col, char in enumerate(line.split()):
                for position in hardware.grid[row][col]:
                    if position in occupied_positions:
                        continue
                    occupied_positions.add(position)
                    break
                else:
                    raise ValueError(f"Cannot find available positions ({row}, {col}) for character {char}. Occupied positions: {hardware.grid[row][col]!r}")
                keys.append(LayoutKey(char, row, col, position.x, position.y, position.finger, position))
            
            row += 1

        if not name:
            name = ''.join(key.char for key in keys[:6])

        return cls(keys, hardware, name)



if __name__ == "__main__":
    char_grid = KeyboardLayout.from_name('qwerty')
    print(repr(char_grid))
    print(str(char_grid))

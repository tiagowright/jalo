import numpy as np
from collections import defaultdict
from typing import List
import os
import re

from hardware import Finger, FingerType, Hand, KeyboardHardware, Position
from dataclasses import dataclass

DEFAULT_HARDWARE = 'ansi'

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
        '''
        Show the keyboard layout in a human-readable format. Prints a neat grid of the layout.
        '''
        COL_SEP = ' '
        HAND_SEP = ' '
        BLANK_KEY = ' '
        
        # identify the longest char in the layout
        longest_char = max(sum(len(key.char) for key in self.grid[row][col]) for row in self.grid for col in self.grid[row])
        
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
    def hardware_hint(cls, name: str) -> str | None:
        text_grid = cls._read_file(name)
        return cls._hardware_hint_in_text(text_grid)

    @classmethod
    def from_name(
        cls, 
        name: str, 
        hardware: KeyboardHardware | None = None, 
        default_hardware: KeyboardHardware | None = None
    ) -> 'KeyboardLayout':
        """
        Load a keyboard layout by name from ``layouts/``.
        
        hardware: specify the hardware that this layout must map to
        default_hardware: use the layout's specified hardware, but if none are specified by the layout, then use this default

        layouts are specified as a comment with a `use:` hint, e.g.
        # use: ansi
        """
        text_grid = cls._read_file(name)
        return cls.from_text(text_grid, name, hardware, default_hardware)


    @classmethod
    def _read_file(cls, name: str) -> str:
        text_file_path = os.path.join('layouts', f'{name}.kb')
        with open(text_file_path, 'r') as file:
            text_grid = file.read()
        return text_grid


    @classmethod
    def _hardware_hint_in_text(cls, text_grid: str):
        hint_re = re.compile(r'\s*#\s*use:\s*([a-z][a-z0-9_]*)')
        for line in text_grid.split('\n'):
            match = hint_re.match(line)
            if match:
                return match.group(1)
        return None


    @classmethod
    def from_text(
        cls, 
        text_grid: str, 
        name: str,
        hardware: KeyboardHardware | None = None, 
        default_hardware: KeyboardHardware | None = None
    ) -> 'KeyboardLayout':
        """Create a keyboard layout from a text grid."""

        if hardware is None:
            hardware_hint = cls._hardware_hint_in_text(text_grid)
            if hardware_hint:
                hardware = KeyboardHardware.from_name(hardware_hint)
            elif default_hardware is not None:
                hardware = default_hardware
            else:
                raise ValueError(f"No hardware specified for layout {name}")


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

            for col_chars in line.split():
                try:
                    col = next(hardware_cols)
                except StopIteration:
                    raise ValueError(f"Too many columns in text grid. Expected {len(hardware.grid[row])}, got {len(line.split())}")

                if len(col_chars) != len(hardware.grid[row][col]):
                    raise ValueError(f"Could not assign all characters and positions for row {row} col {col}: {len(col_chars)} characters '{col_chars}' and {len(hardware.grid[row][col])} positions.")

                for char, position in zip(col_chars, hardware.grid[row][col]):
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

    def swap(self, char_pairs: list[tuple[str, str]], new_name: str = '') -> 'KeyboardLayout':
        '''
        returns a new layout that is the result of swapping the positions of the given character pairs
        '''        
        new_name = new_name or f'{self.name} swapped'

        map_positions = {}
        for char1, char2 in char_pairs:
            if char1 not in self.char_to_key:
                raise ValueError(f"Character {char1} not found in layout")
            if char2 not in self.char_to_key:
                raise ValueError(f"Character {char2} not found in layout")

            pos1 = map_positions.get(self.char_to_key[char1].position, self.char_to_key[char1].position)
            pos2 = map_positions.get(self.char_to_key[char2].position, self.char_to_key[char2].position)

            map_positions[self.char_to_key[char1].position] = pos2
            map_positions[self.char_to_key[char2].position] = pos1
        
        return KeyboardLayout(
            [
                LayoutKey.from_position(map_positions.get(key.position, key.position), key.char)
                for key in self.keys
            ],
            self.hardware, new_name
        )

    def mirror(self, mirrored_name = '') -> 'KeyboardLayout':
        '''
        returns a new layout that is the result of mirroring this layout

        mirroring swaps left and right hand assignments (where possible)
        '''
        # position is a mirror of the other if it has the same row and layer and finger type
        # but hand is the opposite, and column sequence is reversed for that finger
        # finger_to_positions

        mirrored_name = mirrored_name or f'{self.name} mirrored'

        position_grid = {}
        home_position = {}
        for position in self.hardware.positions:
            if position.finger not in position_grid:
                position_grid[position.finger] = {}
                home_position[position.finger] = {}
            if position.col not in position_grid[position.finger]:
                position_grid[position.finger][position.col] = {}
            if position.row not in position_grid[position.finger][position.col]:
                position_grid[position.finger][position.col][position.row] = {}
            if position.layer not in position_grid[position.finger][position.col][position.row]:
                position_grid[position.finger][position.col][position.row][position.layer] = []
            position_grid[position.finger][position.col][position.row][position.layer].append(position)
            if position.is_home:
                home_position[position.finger] = position
        
        finger_map = {
            hand: {
                # notate that the value is of type Finger | None
                fingertype: Finger(0)
                for fingertype in FingerType
            }
            for hand in Hand
        }

        map_positions = {}

        for finger in Finger:
            finger_map[finger.hand][finger.type] = finger

        for fingertype in FingerType:
            lf = finger_map[Hand.LEFT][fingertype]
            rf = finger_map[Hand.RIGHT][fingertype]
            
            # home columns match, then align them in reverse sequence
            if lf not in home_position or rf not in home_position or lf not in position_grid or rf not in position_grid:
                continue

            lh = home_position[lf].col
            rh = home_position[rf].col
            
            for lcol in position_grid[lf]:
                offset = lcol - lh

                # rcol is reversed, so a positive offset from home column on the left is a negative offset on the right
                rcol = rh - offset

                if rcol not in position_grid[rf]:
                    continue
                
                for row in position_grid[lf][lcol]:
                    if row not in position_grid[rf][rcol]:
                        continue
                    for layer in position_grid[lf][lcol][row]:                
                        if layer not in position_grid[rf][rcol][row]:
                            continue
                        for lposition, rposition in zip(position_grid[lf][lcol][row][layer], position_grid[rf][rcol][row][layer]):
                            map_positions[lposition] = rposition
                            map_positions[rposition] = lposition
        
        
        return KeyboardLayout(
            [
                LayoutKey.from_position(
                    map_positions.get(key.position, key.position),
                    key.char
                )
                for key in self.keys
            ],
            self.hardware,
            mirrored_name
        )  
                



if __name__ == "__main__":

    hw = KeyboardHardware.from_name('ansi')
    layout = KeyboardLayout.from_name('qwerty', hw)
    print(layout.name)
    print(str(layout))
    print()

    hw = KeyboardHardware.from_name('ortho')
    layout = KeyboardLayout.from_name('qwerty', hw)
    print(layout.name)
    print(str(layout))
    print()

    ## try using hints
    layout = KeyboardLayout.from_name('hdneu')
    print(f"{layout.name} on {layout.hardware.name}")
    print(str(layout))
    print()

    layout = KeyboardLayout.from_name('inrolly')
    print(f"{layout.name} on {layout.hardware.name}")
    print(str(layout))
    print()

    layout = KeyboardLayout.from_name('hdgold')
    print(f"{layout.name} on {layout.hardware.name}")
    print(str(layout))
    print()
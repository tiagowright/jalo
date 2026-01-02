'''
These classes define the physical attributes of keyboard hardware
'''

from dataclasses import dataclass
from enum import Enum, unique
from re import I
from functools import cache
from typing import List
from collections import defaultdict
import os

@unique
class Hand(Enum):
    """
    Represent the hand of a keyboard user.
    """
    LEFT = 0
    RIGHT = 1

@unique
class FingerType(Enum):
    """
    Represent the type of a finger.
    """
    PINKY = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB = 4

@unique
class Finger(Enum):
    """
    Represent the finger of a keyboard key.
    """
    LP = 0
    LR = 1
    LM = 2
    LI = 3
    LT = 4

    RT = 5
    RI = 6
    RM = 7
    RR = 8
    RP = 9


    @property
    @cache
    def hand(self) -> Hand:
        """Return the hand that owns this finger (cached per enum value)."""
        return Hand(self.value // 5)
        
    @property
    @cache
    def type(self) -> FingerType:
        """
        Return the type of the finger.
        """
        if self.value < 5:
            return FingerType(self.value)

        return FingerType(9 - self.value)
    

@dataclass(frozen=True)
class Position:
    """
    Represent the physical and logical position of a keyboard key.

    Attributes
    ----------
    row : int
        The logical row index of the key.
    col : int
        The logical column index of the key.
    x : float
        The X-coordinate of the key in physical units (e.g., millimeters).
    y : float
        The Y-coordinate of the key in physical units (e.g., millimeters).
    finger : Finger
        The finger assigned to press the key (domain-specific encoding).
    """
    row: int
    col: int
    x: float
    y: float
    finger: Finger
    layer: int = 0
    is_home: bool = False
    heat: float = 0.0

    def __eq__(self, other: 'Position') -> bool:
        return self.row == other.row and self.col == other.col and self.finger == other.finger and self.layer == other.layer

    def __hash__(self) -> int:
        return hash((self.row, self.col, self.finger, self.layer))


class KeyboardHardware:
    """
    Represent the physical layout of a keyboard.

    Attributes
    ----------
    positions : List[Position]
        The positions of the keys on the keyboard.
    rows : List[int]
        The rows of the keyboard.
    cols : List[int]
        The columns of the keyboard.
    finger_to_positions : Dict[Finger, List[Position]]
        The positions of the keys assigned to each finger.
    """
    def __init__(self, name: str, positions: List[Position]):
        self.name = name
        self.positions = sorted(positions, key=lambda x: (x.row, x.col))

        self.rows = list(set(position.row for position in self.positions))
        self.cols = list(set(position.col for position in self.positions))

        self.finger_to_positions = defaultdict(list)
        self.grid = defaultdict(lambda: defaultdict(list))
        for position in self.positions:
            self.finger_to_positions[position.finger].append(position)
            self.grid[position.row][position.col].append(position)

    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_name(cls, name: str) -> 'KeyboardHardware':
        """
        Create a KeyboardHardware instance from a name.
        
        Parameters
        ----------
        name : str
            The name of the keyboard hardware module in the keebs directory.
        
        Returns
        -------
        KeyboardHardware
            The keyboard hardware instance from the specified module.
        """
        # programtically find the file/module with the specified name then return the KEYBOARD element from it
        # the file/module is in the keebs directory
        keebs_dir = os.path.join('keebs', name)
        import importlib
        module = importlib.import_module('keebs.' + name)
        return module.KEYBOARD


    def str(self, show_finger_numbers: bool = False, show_finger_names: bool = False, show_stagger: bool = False) -> str:
        COL_SEP = ' '
        HAND_SEP = ' '
        BLANK_KEY = ' '
        
        min_row = min(self.rows)
        max_row = max(self.rows)
        min_col = min(self.cols)
        max_col = max(self.cols)

        first_col_at_row = {
            row: min(col for col in self.grid[row].keys())
            for row in self.grid.keys()
        }

        stagger_at_row = {
            row: int((self.grid[row][first_col_at_row[row]][0].x - self.grid[row][first_col_at_row[row]][0].col)*100)
            for row in self.grid.keys()
        }

        list_staggers = sorted(set(stagger_at_row.values()))
        
        stagger_strs = {
            row: ' ' * list_staggers.index(stagger_at_row[row])
            for row in self.grid.keys()
        }
        
        def _str_position(position: Position) -> str:
            s = ''
            if show_finger_numbers:
                s += f'{position.finger.value:1d}'
            if show_finger_names:
                s += f'{position.finger.name}'
            if not s:
                s = '.'
            return s

        position_strs = {
            row: {
                col: ''.join(_str_position(position) for position in self.grid[row][col])
                for col in self.grid[row].keys()
            }
            for row in self.grid.keys()
        }

        position_len = max(len(position_strs[row][col]) for row in position_strs.keys() for col in position_strs[row].keys())
        format_str = f'{{:<{position_len}}}'

        text_grid_lines = []
        for row in range(min_row, max_row + 1):
            row_str = []

            if show_stagger:
                row_str.append(stagger_strs[row])

            prev_hand = None
            hand = None
            for col in range(min_col, max_col + 1):

                if row in self.grid and col in self.grid[row]:
                    hand = self.grid[row][col][0].finger.hand

                if hand is not None and prev_hand is not None and hand != prev_hand:
                    row_str.append(HAND_SEP)
                
                prev_hand = hand

                if row in position_strs and col in position_strs[row]:
                    row_str.append(format_str.format(position_strs[row][col]))
                else:
                    row_str.append(format_str.format(BLANK_KEY))

            text_grid_lines.append(COL_SEP.join(row_str))

        return '\n'.join(text_grid_lines)




if __name__ == "__main__":
    # iterate through every file in ./keebs that ends in .py and create a keyboard hardware using the file name
    for file in sorted(os.listdir('keebs')):
        if file.endswith('.py'):
            name = file[:-3]
            keyboard = KeyboardHardware.from_name(name)
            print(f'{keyboard.name}:')
            print(keyboard.str(show_finger_numbers=True, show_stagger=True))
            print()
            print()
